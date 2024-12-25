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
        let client = ChatClient::from_env("gpt-4o").unwrap();
        let name: Name = client.chat("Who was the first president?").await.unwrap();

        assert_eq!(name.first, "George");
        assert_eq!(name.last, "Washington");
    }
}
