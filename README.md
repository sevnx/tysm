# typed-openai

A strongly-typed Rust client for OpenAI's ChatGPT API that enforces type-safe responses using JSON Schema.

## Features

- Type-safe responses using serde and JSON Schema
- Easy to use ChatClient interface
- Support for system prompts and custom messages
- Automatic environment configuration

## Usage

```rust
use typed_openai::ChatClient;

#[derive(Serialize, Deserialize, JsonSchema)]
struct Response {
    message: String,
    confidence: f32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ChatClient::from_env("gpt-4")?;
    let response: Response = client.chat("Tell me a joke").await?;
    println!("{}", response.message);
    Ok(())
}
```

## License

This project is licensed under the MIT License.
