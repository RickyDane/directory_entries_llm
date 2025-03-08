use kalosm::language::*;
use std::fs;

#[tokio::main]
async fn main() {

    loop {
        // Get user input
        let mut user_input = String::new();
        std::io::stdin()
            .read_line(&mut user_input)
            .expect("Couldn't read user_input from stdin");

        let output = get_pred_files("/Users/rickyperlick/Documents", &user_input).await;
        println!("{}", output);
    }
}

async fn get_pred_files(path: &str, query: &str) -> String {
    // Load the system prompt so the llm knows what it needs to do
    let system_prompt =
        fs::read_to_string("./system_prompt.txt").expect("Couldn't read system prompt file");

    println!("{}", system_prompt);

    // Load the specific model to do the inference with
    let model = Llama::new_chat().await.unwrap();

    let mut chat = model.chat().with_system_prompt(&system_prompt);

    let mut ls_entries: String = String::from("<list>\n");

    let entries = fs::read_dir(path);
    if entries.is_err() {
        return "-".into();
    }

    for entry in entries.unwrap() {
        match entry {
            Ok(entry) => ls_entries.push_str(&format!("{}\n", entry.path().to_str().unwrap())),
            Err(_) => continue,
        }
    }

    ls_entries.push_str("</list>\n");

    // Append the user input
    ls_entries.push_str(&format!("<input>\n{}</input>", query));

    // Print whole input
    println!("{}", ls_entries);

    // Start prediction
    let output = chat(&ls_entries).all_text().await;

    return output;
}
