use rand::Rng;

/// A struct that represents a node in the nn
#[derive(Debug)]
pub struct Node {
    /// A vec of the weights associated with the current node.
    /// This is assumed that the length is the same as the amount of nodes in the previous layer.
    weights: Vec<f64>,
    /// A vec of the biases associated with the current node
    /// This is assumed that the length is the same as the amount of nodes in the previous layer.
    biases: Vec<f64>,
}

impl Node {

    /// Method for creating a new `Node`.
    /// Fills unset values with 0's
    pub fn new(capacity: usize, weights_maybe: Option<Vec<f64>>, biases_maybe: Option<Vec<f64>>) -> Self {

        let weights = weights_maybe.unwrap_or(Vec::with_capacity(capacity));
        let biases = biases_maybe.unwrap_or(Vec::with_capacity(capacity));

        assert_eq!(weights.len(), capacity);
        assert_eq!(biases.len(), capacity);

        return Self {
            weights,
            biases,
        };

    }

    /// Method for creating a new `Node` except with randomized initial values set between 0 & 1
    pub fn new_random(capacity: usize) -> Self {

        let mut rng = rand::thread_rng();

        let weights: Vec<f64> = (0..capacity).map(|_| rng.gen_range(0.0..1.0)).collect();
        let biases: Vec<f64> = (0..capacity).map(|_| rng.gen_range(0.0..1.0)).collect();

        return Self {
            weights,
            biases,
        };

    }

}

/// A struct that represents a layer of nodes
/// All layers of nodes are inputs to the following node layer
#[derive(Debug)]
pub struct NodeLayer {
    /// All the nodes on the layer
    pub nodes: Vec<Node>,
    /// The next node layer in the network
    next_node_layer: Option<Box<NodeLayer>>,
}


impl NodeLayer {

    // Method for creating a new `NodeLayer`.
    // Uses the `Node::new_random()` method for initializing its values.
    pub fn new(previous_node_count: usize, capacity: usize) -> Self {

        let mut nodes: Vec<Node> = Vec::with_capacity(capacity);

        for i in 0..nodes.len() {
            println!("{}", i);
            nodes[i] = Node::new_random(previous_node_count);
        }
        println!("Nodes: {:?}", nodes);

        return Self {
            nodes,
            next_node_layer: None,
        };
    }

    /// Method for evaluating the value of the node
    pub fn eval(&self) -> () {
    }
}
