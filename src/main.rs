use ai_in_rust::node::LayerList;

fn main() {

    let amnt_of_nodes_per_layer = 6;
    let amnt_of_layers = 10;

    let layer_list = LayerList::new_random(
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        amnt_of_layers,
        amnt_of_nodes_per_layer,
    );

    let response = layer_list.evaluate();

    println!("Outputs: {:?}", response);

}
