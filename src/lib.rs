mod chatgpt;

pub use chatgpt::ChatClient;
pub use schemars::JsonSchema;
pub use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests {
    use crate::{ChatClient, Deserialize, JsonSchema};

    #[derive(Deserialize, JsonSchema, Debug)]
    struct Name {
        first: String,
        last: String,
    }

    #[tokio::test]
    async fn it_works() {
        let client = ChatClient {
            url: "https://g7edusstdonmn3vxdh3qdypkrq0wzttx.lambda-url.us-east-1.on.aws/v1/chat/completions".to_string(),
            ..ChatClient::from_env("gpt-4o").unwrap()
        };
        let name: Name = client.chat("Who was the first president?").await.unwrap();

        assert_eq!(name.first, "George");
        assert_eq!(name.last, "Washington");

        assert_eq!(client.lru.read().unwrap().len(), 1);
    }
}
