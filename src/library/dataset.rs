use crate::node::{NodeAble, InputLayer, OutputLayer};

/// This struct represents a dataset for teaching the ai model with
pub struct NumberDataSet<T, O>
where T: NodeAble<T>,
      O: NodeAble<O>
{
    /// A vec that represents the data points
    /// This includes input values and the output
    datapoints: Vec<(InputLayer<T>, OutputLayer<O>)>,
}

impl<T: NodeAble<T>, O: NodeAble<O>> NumberDataSet<T, O> {

    /// Creates a new instance of `DataSet`
    pub fn new(datapoints: Vec<(InputLayer<T>, OutputLayer<O>)>) -> Self {
        return Self {
            datapoints,
        };
    }

}

