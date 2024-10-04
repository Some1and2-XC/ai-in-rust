mod lib;

use crate::lib::node::{Node, NodeLayer};

fn main() {

    let layer = NodeLayer::new(5, 5);


    println!("\tThe layer node values:");
    for node in &layer.nodes {
        println!("The layer: `{:?}`", node);
    }

}
