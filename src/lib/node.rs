use std::sync::Arc;
use rand::Rng;

/// A struct that represents a node in the nn
#[derive(Debug)]
pub struct Node<T> {
    /// A vec of the weights associated with the current node.
    /// This is assumed that the length is the same as the amount of nodes in the previous layer.
    weights: Vec<T>,
    /// A vec of the biases associated with the current node
    /// This is assumed that the length is the same as the amount of nodes in the previous layer.
    biases: Vec<T>,
}

pub type StaticNode = f64;

pub enum NodeTypes {
    Node(Node<f64>),
    StaticNode(StaticNode),
}

impl<T> Node<T> {

    /// Method for creating a new `Node`.
    /// Fills unset values with 0's
    pub fn new(capacity: usize, weights_maybe: Option<Vec<T>>, biases_maybe: Option<Vec<T>>) -> Self {

        let weights = weights_maybe.unwrap_or(Vec::with_capacity(capacity));
        let biases = biases_maybe.unwrap_or(Vec::with_capacity(capacity));

        assert_eq!(weights.len(), capacity);
        assert_eq!(biases.len(), capacity);

        return Self {
            weights,
            biases,
        };

    }
}

impl Node<f64> {

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

impl Node<f64> {

    pub fn evalute(&self, input: &Vec<f64>) -> f64 {

        let mut output: f64 = 0.0;
        let output_count = input.len();

        assert_eq!(output_count, input.len());

        for i in 0..output_count {

            output += self.weights[i] * input[i] + self.biases[i];

        }

        return output / (output_count as f64);

    }

}

/// A struct that represents a layer of nodes
/// All layers of nodes are inputs to the following node layer
#[derive(Debug)]
pub struct NodeLayer<T> {
    /// All the nodes on the layer
    pub nodes: Vec<Node<T>>,
    /// The next node layer in the network
    next_node_layer: Option<Arc<NodeLayer<T>>>,
}

impl<T> NodeLayer<T> {

    /// Method for creating a new `NodeLayer` from an array of nodes
    pub fn new_from_nodes(nodes: Vec<Node<T>>) -> Self {

        return Self {
            nodes,
            next_node_layer: None,
        };

    }
}

impl NodeLayer<f64> {

    /// Method for creating a new `NodeLayer`.
    /// Uses the `Node::new_random()` method for initializing its values.
    pub fn new_random(previous_node_count: usize, capacity: usize, next_node_layer: Option<Arc<NodeLayer<f64>>>) -> Self {

        let mut nodes: Vec<Node<f64>> = Vec::with_capacity(capacity);

        for i in 0..capacity {
            nodes.push(Node::new_random(previous_node_count));
        }

        return Self {
            nodes,
            next_node_layer,
        };
    }

    /// Method for evaluating the value of the node
    pub fn evaluate(&self, input: &Vec<f64>) -> Vec<f64> {

        // Initializes output
        let mut output = Vec::with_capacity(self.nodes.len());

        // Calculates all nodes and adds result to output
        for node in &self.nodes {
            output.push(node.evalute(input));
        }

        // Returns output
        return output;

    }
}

#[derive(Debug)]
pub struct InputLayer<T> {
    /// All the nodes on the layer
    pub nodes: Vec<T>,
    /// The next node layer in the network
    next_node_layer: Option<Arc<NodeLayer<T>>>,
}

impl<T> InputLayer<T> {

    /// Creates a new `InputLayer`
    pub fn new(nodes: Vec<T>, next_node_layer: Option<Arc<NodeLayer<T>>>) -> Self {

        return Self {
            nodes,
            next_node_layer,
        };

    }

}

impl InputLayer<f64> {

    pub fn evaluate(&self) -> Vec<f64> {

        // Returns early if not found
        let mut next_layer = match &self.next_node_layer {
            Some(v) => v.clone(),
            None => return self.nodes.clone(), // we don't mind the clone here because it shouldn't
                                               // really happen unless the user is doing some
                                               // weird sh*t
        };

        let mut data: Vec<f64>;

        loop {

            data = next_layer.evaluate(&self.nodes);
            next_layer = match &next_layer.next_node_layer {
                Some(v) => v.clone(),
                None => break,
            };

        }

        return data;

    }


}
