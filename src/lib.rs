mod chatgpt;

pub use chatgpt::ChatClient;
pub use schemars::JsonSchema;
pub use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests {
    use crate::{ChatClient, Deserialize, JsonSchema};

    use std::sync::LazyLock;
    static CLIENT: LazyLock<ChatClient> = LazyLock::new(|| {
        let my_api = "https://g7edusstdonmn3vxdh3qdypkrq0wzttx.lambda-url.us-east-1.on.aws/v1/chat/completions".to_string();
        ChatClient {
            url: my_api,
            ..ChatClient::from_env("gpt-4o").unwrap()
        }
    });

    #[derive(Deserialize, JsonSchema, Debug)]
    struct Name {
        first: String,
        last: String,
    }

    #[tokio::test]
    async fn it_works() {
        let name: Name = CLIENT.chat("Who was the first president?").await.unwrap();

        assert_eq!(name.first, "George");
        assert_eq!(name.last, "Washington");

        assert_eq!(CLIENT.lru.read().unwrap().len(), 1);

        let usage = CLIENT.usage();
        for _ in 0..5 {
            let _name: Name = CLIENT.chat("Who was the first president?").await.unwrap();
        }
        let usage2 = CLIENT.usage();
    }
}
