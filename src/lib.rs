mod chatgpt;

pub use chatgpt::ChatClient;

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use chatgpt::ChatClient;
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    use super::*;

    #[derive(Serialize, Deserialize, JsonSchema, Debug)]
    struct Name {
        first: String,
        last: String,
    }

    #[tokio::test]
    async fn it_works() {
        let client = ChatClient::from_env("gpt-4o").unwrap();
        let name: Name = client.chat("Who was the first president?").await.unwrap();

        assert_eq!(name.first, "George");
        assert_eq!(name.last, "Washington");
    }
}
