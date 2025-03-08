use kalosm::language::*;
use std::fs;

#[tokio::main]
async fn main() {
    let system_prompt =
        fs::read_to_string("./system_prompt.txt").expect("Couldn't read system prompt file");

    let mut ls_entries: String = String::from("<list>\n");

    let entries = fs::read_dir("/Users/rickyperlick/Documents");
    if entries.is_err() {
        return;
    }

    for entry in entries.unwrap() {
        match entry {
            Ok(entry) => ls_entries.push_str(&format!("{}\n", entry.path().to_str().unwrap())),
            Err(_) => continue,
        }
    }

    ls_entries.push_str("</list>\n");

    println!("Wonach wird gesucht? > ");

    let mut user_input: String = String::new();
    std::io::stdin().read_line(&mut user_input).expect("Couldn't read user_input from stdin");
    ls_entries.push_str(&format!("<input>\n{}</input>", user_input));

    println!("{}", ls_entries);

    let model = Llama::builder()
            .with_source(LlamaSource::llama_3_2_3b_chat())
            .build()
            .await
            .unwrap();
    let mut chat = model.chat().with_system_prompt(system_prompt);

    chat(&ls_entries).to_std_out().await.expect("chat didn't go as planned");
}
