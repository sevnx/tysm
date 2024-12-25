mod chatgpt;

pub use chatgpt::ChatClient;

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use chatgpt::{AiResponse, ChatClient};
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    use super::*;

    #[derive(Serialize, Deserialize, JsonSchema, Debug)]
    struct Name {
        first: String,
        last: String,
    }

    impl AiResponse for Name {
        const NAME: &'static str = "FullName";
    }

    #[tokio::test]
    async fn it_works() {
        let client = ChatClient::from_env("gpt-4o").unwrap();
        let result: Name = client.chat("Who was the first president?").await.unwrap();

        assert_eq!(result.first, "George");
        assert_eq!(result.last, "Washington");
    }
}
