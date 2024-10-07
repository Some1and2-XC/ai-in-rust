mod lib;

use crate::lib::node::{NodeLayer, InputLayer};

fn main() {

    let amnt_of_nodes_per_layer = 6;
    let amnt_of_layers = 10;

    let input_nodes = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // Initializes `layers`
    let mut layers: Vec<NodeLayer<f64>> = Vec::with_capacity(amnt_of_layers);

    layers.push(NodeLayer::new_random(amnt_of_nodes_per_layer, amnt_of_nodes_per_layer, None));

    for i in (amnt_of_layers-1)..=1 {

        let prev_count = match i == 1 {
            true => input_nodes.len(),
            false => amnt_of_nodes_per_layer,
        };

        layers.push(NodeLayer::new_random(prev_count, amnt_of_nodes_per_layer, Some(&layers[i-1])));

    }

    // Makes sure there is at least 1 value in the array
    assert_ne!(layers.len(), 0);

    let mut input = InputLayer::new(input_nodes, Some(&layers[0]));

    // Sets the input nodes "next" as the first value
    input.next_node_layer = Some(&layers[0]);

    let response = input.evaluate();

    println!("Outputs: {:?}", response);

}
