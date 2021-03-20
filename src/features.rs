extern crate serde;
use serde::{Serialize, Deserialize};
use std::fs;

#[derive(Debug, Deserialize, Serialize)]
pub struct Feature {
    length: u16,
    width: u16,
} 
