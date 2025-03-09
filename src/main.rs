use kalosm::language::*;
use std::{fs, path::Path};

#[tokio::main]
async fn main() {
    loop {
        // Get user input
        let mut user_input = String::new();
        std::io::stdin()
            .read_line(&mut user_input)
            .expect("Couldn't read user_input from stdin");

        do_task("/Users/rickyperlick/Documents", &user_input).await;

        //     let output = get_pred_files("/Users/rickyperlick/Documents", &user_input).await;
        //     println!("Final result:\n{}", output);
    }
}

#[derive(Parse, Schema, Clone, Debug)]
struct DirectoryPathEntry {
    path: String,
}

async fn do_task(path: &str, query: &str) -> String {
    // Load the system prompt so the llm knows what it needs to do
    let system_prompt =
        fs::read_to_string("./system_prompt.txt").expect("Couldn't read system prompt file");

    // Load the specific model to do the inference with
    let model = Llama::builder()
        .with_source(LlamaSource::solar_10_7b_instruct())
        .build()
        .await
        .unwrap();

    // Then create a task with the parser as constraints
    let task = model.task(system_prompt).typed();

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

    println!("{}", ls_entries);

    // Finally, run the task
    let mut stream = task(&ls_entries);
    stream.to_std_out().await.unwrap();

    let paths: Vec<DirectoryPathEntry> = stream.await.unwrap();

    println!("\nResults:");

    for path in paths {
        if Path::new(&path.path).exists() {
            println!("{}", path.path);
        } else {
            println!("{} does not exist", path.path);
        }
    }

    return "".into();
}

async fn get_pred_files(path: &str, query: &str) -> String {
    // Load the system prompt so the llm knows what it needs to do
    let system_prompt =
        fs::read_to_string("./system_prompt.txt").expect("Couldn't read system prompt file");

    // Load the specific model to do the inference with
    let model = Llama::builder()
        .with_source(LlamaSource::open_chat_7b())
        // .with_source(
        //     LlamaSource::new(FileSource::local(PathBuf::from("./gemma2-2b-it-Q3.gguf")))
        //         .with_tokenizer(FileSource::local(PathBuf::from("./tokenizer.json"))),
        // )
        // .with_source(LlamaSource::new(FileSource::local(PathBuf::from("./qwen1_5-4b-chat-q5_0.gguf"))).with_tokenizer(FileSource::local(PathBuf::from("./tokenizer-2.json"))))
        .with_source(
            LlamaSource::new(FileSource::huggingface(
                "TheBloke/stablelm-zephyr-3b-GGUF",
                "main",
                "stablelm-zephyr-3b.Q2_K.gguf",
            ))
            .with_tokenizer(FileSource::huggingface(
                "stabilityai/stablelm-zephyr-3b",
                "main",
                "tokenizer.json",
            )),
        )
        .build()
        .await
        .unwrap_or_else(|x| panic!("Couldn't load model: {}", x));

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

    println!("{}", ls_entries);

    // Start prediction
    let output = chat(&ls_entries).all_text().await;

    println!("[DEBUG] {}", output);

    return output
        .split("<result>")
        .nth(1)
        .unwrap_or("")
        .split("</result>")
        .nth(0)
        .unwrap_or("")
        .into();
}
