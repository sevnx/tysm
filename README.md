# tysm

A strongly-typed Rust client for OpenAI's ChatGPT API that enforces type-safe responses using [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs).

## Features

- Type-safe API responses
- Easy to use ChatClient interface
- Support for system prompts and custom messages
- Automatically get an OpenAI API Key from the environment

## Usage

```rust
use tysm::{
    ChatClient, 
    Deserialize, // re-exported from `serde` for convenience
    JsonSchema,  // re-exported from `schemars` for convenience
};

#[derive(Deserialize, JsonSchema)]
struct Name {
    first: String,
    last: String,
}

async fn typed_name() {
    let client = ChatClient::from_env("gpt-4o").unwrap();
    let name: Name = client.chat("Who was the first president?").await.unwrap();

    assert_eq!(name.first, "George");
    assert_eq!(name.last, "Washington");
}
```

## License

This project is licensed under the MIT License.
