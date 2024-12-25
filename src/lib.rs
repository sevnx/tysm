mod chatgpt;

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use chatgpt::AiType;
    use schemars::{schema_for, JsonSchema};
    use serde::{Deserialize, Serialize};

    use super::*;

    #[derive(Serialize, Deserialize, JsonSchema, Debug)]
    struct Name {
        first: String,
        last: String,
    }

    impl AiType for Name {
        const NAME: &'static str = "FullName";
    }

    #[tokio::test]
    async fn it_works() {
        let result = chatgpt::call::<Name>(
            "gpt-4o".to_string(),
            "Who was the first president?".to_string(),
            "".to_string(),
        )
        .await
        .unwrap();
        panic!("{result:?}");
    }
}
