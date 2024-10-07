use std::{fmt::{Display, Debug}, ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign}};

use rand::{thread_rng, Rng};

/// This trait is used for specifying what can be used as the base type in a node.
pub trait NodeAble<T>:
Add<Output = T> + AddAssign +
Sub<Output = T> + SubAssign +
Mul<Output = T> + MulAssign +
Div<Output = T> + DivAssign +
Sized + Sync +
Clone + Copy +
Debug + Display {
    /// Randomly generates a number between 0 and 1
    fn get_random() -> Self;
    /// Gets a value from an f64
    fn from_f64(n: f64) -> Self;
    /// Gets a value from a usize
    fn from_usize(n: usize) -> Self;
    /// Gets a zeroed value
    fn new_zero() -> Self;
}

impl NodeAble<f64> for f64 {
    fn get_random() -> Self {
        let mut rng = thread_rng();
        return rng.gen_range(0.0..1.0);
    }

    fn from_f64(n: f64) -> Self { return n as Self; }
    fn new_zero() -> Self { return 0.0; }
    fn from_usize(n: usize) -> Self { return n as Self; }

}

impl NodeAble<f32> for f32 {
    fn get_random() -> Self {
        let mut rng = thread_rng();
        return rng.gen_range(0.0..1.0);
    }

    fn from_f64(n: f64) -> Self { return n as Self; }
    fn new_zero() -> Self { return 0.0; }
    fn from_usize(n: usize) -> Self { return n as Self; }
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

        let weights: Vec<T> = (0..capacity).map(|_| T::get_random()).collect();
        let biases: Vec<T> = (0..capacity).map(|_| T::get_random()).collect();

        return Self {
            weights,
            biases,
        };

    }

    /// Method for evaluating a node from the input vec `input`
    pub fn evalute(&self, input: &Vec<T>) -> T {

        let mut output = T::new_zero();
        let output_count = input.len();

        assert_eq!(output_count, input.len());

        for i in 0..output_count {
            output += self.weights[i] * input[i] + self.biases[i];
        }

        return output / T::from_usize(output_count);

    }

}

/// A struct that represents a layer of nodes
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

/// A type representing a layer of input nodes
pub type InputLayer<T> where T: NodeAble<T> = Vec<T>;
/// A type representing a layer of output nodes
pub type OutputLayer<T> where T: NodeAble<T> = Vec<T>;

/// A struct representing an entire network.
/// This struct holds both an input layer and multiple evaluation layers.
pub struct LayerList<T>
where T: NodeAble<T> {
    /// This is the input layer that gets fed into the network.
    pub input_layer: InputLayer<T>,
    /// These are the inner layers or "hidden layers" that make up the network.
    pub layers: Vec<NodeLayer<T>>,
}


impl<T: NodeAble<T>> LayerList<T> {

    /// Method for creating new `LayerList`.
    pub fn new(input_layer: InputLayer<T>, layers: Vec<NodeLayer<T>>) -> Self {
        return Self {
            input_layer,
            layers,
        };
    }

    /// Initializes `LayerList` with random values for all its nodes.
    pub fn new_random(input_layer: InputLayer<T>, amnt_of_layers: usize, node_count: usize) -> Self {

        assert_ne!(amnt_of_layers, 0); // ensures we have at least 1 layer

        // We initialize ourself first
        let mut this = Self {
            input_layer,
            layers: Vec::with_capacity(amnt_of_layers),
        };

        // We then add the first layer outself while specifying out amount of nodes in the input
        this.push(NodeLayer::new_random((&this.input_layer).len(), node_count));

        // And finally we can use a for loop to initialize all other values
        for _i in 1..amnt_of_layers {
            this.push(NodeLayer::new_random(node_count, node_count));
        }

        // Then we return the created `LayerList`
        return this;

    }

    /// Method for adding a NodeLayer to the layers
    pub fn push(&mut self, value: NodeLayer<T>) -> () {
        return self.layers.push(value);
    }

    /// Evaluates the neural network based on its attributes.
    pub fn evaluate(&self) -> OutputLayer<T> {

        // We actually do want to clone here.
        // The tmp_values are going to be set over and over so having an owned vec is ideal.
        let mut tmp_values = self.input_layer.clone();

        for i in &self.layers {
            tmp_values = i.evaluate(&tmp_values);
        }

        return tmp_values;

    }

}
