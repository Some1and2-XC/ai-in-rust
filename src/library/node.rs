use std::ops::{Add, Sub, Mul, Div};

use rand::{Rng, prelude::Distribution};

trait NodeAble<T>: Add + Sub + Mul + Div + Clone + Copy {}

impl NodeAble<f64> for f64 {

}

/// A struct that represents a node in the nn
#[derive(Debug, Clone)]
pub struct Node<T> where T: NodeAble<T> {
    /// A vec of the weights associated with the current node.
    /// This is assumed that the length is the same as the amount of nodes in the previous layer.
    weights: Vec<T>,
    /// A vec of the biases associated with the current node
    /// This is assumed that the length is the same as the amount of nodes in the previous layer.
    biases: Vec<T>,
}

impl<T: NodeAble<T>> Node<T> {

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

    /// Method for creating a new `Node` except with randomized initial values set between 0 & 1
    pub fn new_random(capacity: usize) -> Self {

        let mut rng = rand::thread_rng();

        let weights: Vec<T> = (0..capacity).map(|_| rng.gen()).collect();
        let biases: Vec<T> = (0..capacity).map(|_| rng.gen()).collect();

        return Self {
            weights,
            biases,
        };

    }

    pub fn evalute(&self, input: &Vec<T>) -> f64 {

        let mut output: T = 0.into();
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
#[derive(Debug, Clone)]
pub struct NodeLayer<T> where T: NodeAble<T> {
    /// All the nodes on the layer
    pub nodes: Vec<Node<T>>,
}

impl<T: NodeAble<T>> NodeLayer<T> {

    /// Method for creating a new `NodeLayer` from an array of nodes
    pub fn new_from_nodes(nodes: Vec<Node<T>>) -> Self {

        return Self {
            nodes,
        };

    }

    /// Method for creating a new `NodeLayer`.
    /// Uses the `Node::new_random()` method for initializing its values.
    pub fn new_random(previous_node_count: usize, capacity: usize) -> Self {

        let mut nodes: Vec<Node<T>> = Vec::with_capacity(capacity);

        for _i in 0..capacity {
            nodes.push(Node::new_random(previous_node_count));
        }

        return Self {
            nodes,
        };
    }

    /// Method for evaluating the value of the node
    pub fn evaluate(&self, input: &Vec<T>) -> Vec<T> {

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
}

impl<T> InputLayer<T> {

    /// Creates a new `InputLayer`
    pub fn new(nodes: Vec<T>) -> Self {

        return Self {
            nodes,
        };

    }

}

impl InputLayer<f64> {

}

pub struct LayerList<T>
where T: NodeAble<T> {
    pub layers: Vec<NodeLayer<T>>,
    pub input_layer: InputLayer<T>,
}


impl<T: NodeAble<T>> LayerList<T> {

    /*
    pub fn evaluate(&self) -> Vec<T> {

        let tmp_values: Vec<&T> = self.input_layer.nodes.iter()
            .map(|v| v)
            .collect();

        loop {

            tmp_values = next_layer.evaluate(&self.nodes);
            next_layer = match &next_layer.next_node_layer {
                Some(v) => v.clone(),
                None => break,
            };

        }

        return tmp_values;

    }
    */

}
