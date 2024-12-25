use reqwest::Client;
use schemars::{schema::RootSchema, schema_for, JsonSchema};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use thiserror::Error;

pub struct ChatClient {
    api_key: String,
    url: String,
    model: String,
}

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
    #[expect(unused)]
    id: String,
    #[expect(unused)]
    object: String,
    #[expect(unused)]
    created: u64,
    #[expect(unused)]
    model: String,
    #[expect(unused)]
    system_fingerprint: Option<String>,
    choices: Vec<ChatChoice>,
    #[expect(unused)]
    usage: ChatUsage,
}

#[derive(Deserialize, Debug)]
struct ChatChoice {
    #[expect(unused)]
    index: u8,
    message: ChatMessage,
    #[expect(unused)]
    logprobs: Option<serde_json::Value>,
    #[expect(unused)]
    finish_reason: String,
}

#[derive(Deserialize, Debug)]
struct ChatUsage {
    #[expect(unused)]
    prompt_tokens: u32,
    #[expect(unused)]
    completion_tokens: u32,
    #[expect(unused)]
    total_tokens: u32,
}

fn api_key() -> Result<String, std::env::VarError> {
    use dotenv::dotenv;
    dotenv().ok();
    std::env::var("OPENAI_API_KEY")
}

#[derive(Error, Debug)]
pub enum ChatError {
    #[error("API key not found: {0}")]
    ApiKeyNotFound(#[from] std::env::VarError),

    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    #[error("JSON parsing error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("No choices returned from API")]
    NoChoices,
}

impl ChatClient {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            url: "https://api.openai.com/v1/chat/completions".into(),
            model: model.into(),
        }
    }

    pub fn from_env(model: impl Into<String>) -> Result<Self, std::env::VarError> {
        Ok(Self::new(api_key()?, model))
    }

    pub async fn chat<T: DeserializeOwned + JsonSchema>(
        &self,
        prompt: impl Into<String>,
    ) -> Result<T, ChatError> {
        self.chat_with_system_prompt(prompt, "").await
    }

    pub async fn chat_with_system_prompt<T: DeserializeOwned + JsonSchema>(
        &self,
        prompt: impl Into<String>,
        system_prompt: impl Into<String>,
    ) -> Result<T, ChatError> {
        let prompt = prompt.into();
        let system_prompt = system_prompt.into();

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
        self.chat_with_messages::<T>(messages).await
    }

    pub async fn chat_with_messages<T: DeserializeOwned + JsonSchema>(
        &self,
        messages: Vec<ChatMessage>,
    ) -> Result<T, ChatError> {
        let client = Client::new();

        let schema = schema_for!(T);

        let chat_request = ChatRequest {
            model: self.model.clone(),
            messages,
            response_format: ResponseFormat {
                format_type: "json_schema".to_string(),
                json_schema: JsonSchemaFormat {
                    name: tynm::type_name::<T>(),
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
            .post(self.url.clone())
            .header("Authorization", format!("Bearer {}", self.api_key.clone()))
            .header("Content-Type", "application/json")
            .json(&chat_request)
            .send()
            .await?;

        let chat_response = response.text().await?;
        let chat_response: ChatResponse = serde_json::from_str(&chat_response)?;
        let chat_response = chat_response
            .choices
            .first()
            .ok_or(ChatError::NoChoices)?
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

        let chat_response: T = serde_json::from_str(&chat_response)?;

        Ok(chat_response)
    }
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
    let _chat_response: ChatResponse = serde_json::from_str(&s).unwrap();
}
