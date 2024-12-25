use anyhow::{anyhow, bail, Context, Result};
use reqwest::Client;
use schemars::{schema::RootSchema, schema_for, JsonSchema};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub enum Role {
    #[serde(rename = "user")]
    User,
    #[serde(rename = "assistant")]
    Assistant,
    #[serde(rename = "system")]
    System,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    response_format: ResponseFormat,
}

#[derive(Serialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    pub json_schema: JsonSchemaFormat,
}

#[derive(Serialize)]

pub struct JsonSchemaFormat {
    name: String,
    strict: bool,
    schema: SchemaFormat,
}

#[derive(Serialize)]
pub struct SchemaFormat {
    #[serde(rename = "additionalProperties")]
    additional_properties: bool,
    #[serde(flatten)]
    schema: RootSchema,
}

#[derive(Deserialize, Debug)]
struct ChatResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    system_fingerprint: Option<String>,
    choices: Vec<ChatChoice>,
    usage: ChatUsage,
}

#[derive(Deserialize, Debug)]
struct ChatChoice {
    index: u8,
    message: ChatMessage,
    logprobs: Option<serde_json::Value>,
    finish_reason: String,
}

#[derive(Deserialize, Debug)]
struct ChatUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

pub trait AiType: Serialize + DeserializeOwned + JsonSchema {
    const NAME: &'static str;
}

fn api_key() -> anyhow::Result<String> {
    use dotenv::dotenv;
    dotenv().ok();
    std::env::var("OPENAI_API_KEY").map_err(|e| {
        anyhow!("Expected OPENAI_API_KEY to be set ({e:?}), make sure it's set in dotenv!")
    })
}

pub async fn call<T: AiType>(
    model: String,
    prompt: String,
    system_prompt: String,
) -> anyhow::Result<T> {
    let messages = vec![
        ChatMessage {
            role: Role::System,
            content: system_prompt,
        },
        ChatMessage {
            role: Role::User,
            content: prompt,
        },
    ];
    call_with_messages::<T>(model, messages).await
}

pub async fn call_with_messages<T: AiType>(
    model: String,
    messages: Vec<ChatMessage>,
) -> anyhow::Result<T> {
    let api_key = api_key()?;
    let client = Client::new();
    let api_url = "https://api.openai.com/v1/chat/completions";

    let schema = schema_for!(T);

    let chat_request = ChatRequest {
        model,
        messages,
        response_format: ResponseFormat {
            format_type: "json_schema".to_string(),
            json_schema: JsonSchemaFormat {
                name: T::NAME.to_string(),
                strict: true,
                schema: SchemaFormat {
                    additional_properties: false,
                    schema,
                },
            },
        },
    };

    println!("{}", serde_json::to_string_pretty(&chat_request).unwrap());

    let response = client
        .post(api_url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&chat_request)
        .send()
        .await?;

    let chat_response = response.text().await?;
    let chat_response: ChatResponse =
        serde_json::from_str(&chat_response).context(format!("deserializing {chat_response:?}"))?;
    let chat_response = chat_response
        .choices
        .get(0)
        .ok_or_else(|| anyhow!("Expected at least one choice, got {chat_response:?}"))?
        .message
        .content
        .clone();

    let chat_response = chat_response
        .strip_prefix("```")
        .unwrap_or(&chat_response)
        .strip_prefix("json")
        .unwrap_or(&chat_response)
        .strip_suffix("```")
        .unwrap_or(&chat_response)
        .to_string();

    let chat_response: T =
        serde_json::from_str(&chat_response).context(format!("deserializing {chat_response:?}"))?;

    Ok(chat_response)
}

#[test]
fn test_deser() {
    let s = r#"
{
    "choices": [
        {
        "finish_reason": "stop",
        "index": 0,
        "logprobs": null,
        "message": {
            "content": "Hey there! When replying to someone on a dating app who's asked about what you're studying, it's all about how you present it. Even if you think math might sound boring, you can share why you find it interesting or how it applies to everyday life. Try saying something like, \"I'm actually diving into the world of math! It's fascinating because [insert a fun fact about your studies or why you chose it]. What about you? What are you passionate about?\" This way, you're flipping the script from just stating your major to sharing your enthusiasm, which can be very attractive!",
            "role": "assistant"
        }
        }
    ],
    "created": 1714696172,
    "id": "chatcmpl-9Kb5oqHOdNRLuFJHCTQFOeU516mU8",
    "model": "gpt-4-0125-preview",
    "object": "chat.completion",
    "system_fingerprint": null,
    "usage": {
        "completion_tokens": 123,
        "prompt_tokens": 188,
        "total_tokens": 311
    }
}
"#;
    let chat_response: ChatResponse = serde_json::from_str(&s)
        .context(format!("deserializing {s:?}"))
        .unwrap();
}
