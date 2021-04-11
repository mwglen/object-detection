#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

mod detect;
mod cascade;
mod map;
mod recognize;

extern crate clap;
use clap::{load_yaml, App};

fn main() {
    let yaml = load_yaml!("cli.yml");
    match App::from_yaml(yaml).get_matches().subcommand() {
        ("cascade", Some(m)) => cascade::main(m),
        ("recognize", Some(m)) => recognize::main(m),
        ("detect", Some(m)) => detect::main(m),
        ("map", Some(m)) => map::main(m),
        _ => (),
    }
}
