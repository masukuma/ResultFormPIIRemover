use clap::Parser;

fn main() -> std::process::ExitCode {
    let cli = rfpii_api::cli::Cli::parse();
    rfpii_api::cli::init_logging(&cli);
    match rfpii_api::cli::run(cli) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("rfpii-api: {e:#}");
            std::process::ExitCode::FAILURE
        }
    }
}
