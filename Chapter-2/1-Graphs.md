# Graph Structure of Tensorflow 

## Introduction
- In Tensorflow, We need to define the blueprint of a Neural Network by using a Computational Graph and then write script of it's execution.

## Computational Graphs
- A Computational Graph is a Network of **Nodes** and **Edges**.

- It consists of:
  1) **Tensor Objects**
     - Constants - Statically defined, Their value can't be changed
     - Variables - Statically defined, Their value may be modified after definition
     - Placeholders - A variable whose value can be assigned later
  2) **Operation Objects**
     - They represent the Computations


- A **Node** is a construct represented by **Tensors** and **Operations** which may take multiple inputs, but gives only one output.
- **Edges** are the Tensors which flow in between the operations.

## Execution of the graph
- A **Session Object** holds the environment which Tensor and Operation objects are computed.
- The blueprint given by the Computational Graph is initialized and run here.

## Motivation for using Computational Graphs
- Graphs are an intuitive construct to build Neural Networks.

- **Removing common sub-expressions:**
  - This means that if the same expression is calculated multiple times in a graph, TensorFlow can remove the duplicate calculations and reuse the results. This can improve performance by reducing the number of operations that need to be performed.

- **Fusing kernels:**
  - This means that TensorFlow can combine multiple kernels into a single kernel. This can improve performance by reducing the number of times that the data needs to be transferred between the CPU and the GPU.
  
- **Cutting redundant expressions:**
  - This means that TensorFlow can remove expressions that are not needed. This can improve performance by reducing the number of operations that need to be performed.

## Example: Adding two vectors

1) Defining the Computational Graph
    ```python
    v_1 = tf.constant([1,1,1,1])
    v_2 = tf.constant([1,2,3,4])
    v_add = tf.add(v_1,v_2)
    ```
2) Running the Session
    ```python
    with tf.Session() as sess:
     print(sess.run(v_add))
   
   # Output: [2, 3, 4, 5]
    ```
   