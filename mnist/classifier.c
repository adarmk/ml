#include <stddef.h>
#include <stdlib.h>
#include<stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <assert.h>

const unsigned int IN_LAYER_LEN = 784;
const unsigned int H1_LEN = 128;
const unsigned int H2_LEN = 128; 
const unsigned int OUT_LAYER_LEN = 10; 

const unsigned int BATCH_SIZE = 32;
const unsigned int EPOCHS = 2; // Achieves ~86-90% accuracy after 1 epoch, ~92% after 2 epochs. Beyond that it likely begins to overfit
const float LEARNING_RATE = 0.02;

// Matrix math 
typedef struct {
    float *data;
    unsigned int rows;
    unsigned int cols;
} Matrix;

Matrix alloc_mat(unsigned int rows, unsigned int cols) {
    Matrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.data = (float *) malloc(rows * cols * sizeof(float));
    if (mat.data == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    return mat;
}

void free_mat(Matrix mat) {
    if (mat.data != NULL) {
        free(mat.data);
        mat.data = NULL;
    }
}

static inline float unchecked_index(Matrix *x, unsigned int row, unsigned int col) {
    return x->data[row * x->cols + col];
}

static inline void unchecked_set_value(Matrix *m, unsigned int row, unsigned int col, float value) {
    if (row < m->rows && col < m->cols) {
        m->data[row * m->cols + col] = value;
    } else {
        fprintf(stderr, "Error: Index out of bounds when setting matrix value\n");
        exit(EXIT_FAILURE);
    }
}

Matrix hadamard_product(Matrix a, Matrix b) {
    if (a.rows != b.rows || a.cols != 1 || b.cols != 1) {
        fprintf(stderr, "Error: Matrices must be vectors of the same length\n");
        exit(EXIT_FAILURE);
    }

    Matrix result = alloc_mat(a.rows, 1);

    for (unsigned int i = 0; i < a.rows; i++) {
        float value = unchecked_index(&a, i, 0) * unchecked_index(&b, i, 0);
        unchecked_set_value(&result, i, 0, value);
    }

    return result;
}


Matrix mat_mul(Matrix a, Matrix b) {
    if (a.cols != b.rows) {
        fprintf(stderr, "Error: Matrices have incompatible dimensions for multiplication\n");
        exit(EXIT_FAILURE);
    }

    Matrix prod = alloc_mat(a.rows, b.cols);

    for (unsigned int i = 0; i < a.rows; i++) {
        for (unsigned j = 0; j < b.cols; j++) {
            float sum = 0;
            for (unsigned int k = 0; k < a.cols; k++) {
                sum += unchecked_index(&a, i, k) * unchecked_index(&b, k, j);
            }
            unchecked_set_value(&prod, i, j, sum);
        }
    }

    return prod;
}

Matrix transpose(Matrix m) {
    Matrix m_t = alloc_mat(m.cols, m.rows);

    for (unsigned int i = 0; i < m.cols; i++) {
        for (unsigned j = 0; j < m.rows; j++) {
            unchecked_set_value(&m_t, i, j, unchecked_index(&m, j, i));
        }
    }

    return m_t;
}


Matrix rand_matrix(unsigned int rows, unsigned int cols) {
    Matrix mat = alloc_mat(rows, cols);
    srand(time(NULL)); // Seed the random number generator

    for (unsigned int i = 0; i < rows; i++) {
        for (unsigned int j = 0; j < cols; j++) {
            float random_value = ((float)rand() / RAND_MAX) * 0.2 - 0.1; // Generate random value between -0.1 and 0.1
            unchecked_set_value(&mat, i, j, random_value);
        }
    }

    return mat;
}

Matrix zero_vector(unsigned int n) {
    Matrix vec = alloc_mat(n, 1);

    for (unsigned int i = 0; i < n; i++) {
        unchecked_set_value(&vec, i, 0, 0.0);
    }

    return vec;
}

Matrix mat_add(Matrix a, Matrix b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        printf("Matrix a dimensions: %u rows, %u cols\n", a.rows, a.cols);
        printf("Matrix b dimensions: %u rows, %u cols\n", b.rows, b.cols);
        assert(a.rows == b.rows && a.cols == b.cols);
    }

    Matrix result = alloc_mat(a.rows, a.cols);

    for (unsigned int i = 0; i < a.rows; i++) {
        for (unsigned int j = 0; j < a.cols; j++) {
            float sum = unchecked_index(&a, i, j) + unchecked_index(&b, i, j);
            unchecked_set_value(&result, i, j, sum);
        }
    }

    return result;
}

void mat_add_mut(Matrix *a, Matrix b) {
    
    if (a->rows != b.rows || a->cols != b.cols) {
        printf("Matrix a dimensions: %u rows, %u cols\n", a->rows, a->cols);
        printf("Matrix b dimensions: %u rows, %u cols\n", b.rows, b.cols);
        assert(a->rows == b.rows && a->cols == b.cols);
    }

    for (unsigned int i = 0; i < a->rows; i++) {
        for (unsigned int j = 0; j < a->cols; j++) {
            float sum = unchecked_index(a, i, j) + unchecked_index(&b, i, j);
            unchecked_set_value(a, i, j, sum);
        }
    }
}


Matrix mat_sub(Matrix a, Matrix b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        fprintf(stderr, "Error: Both matrices must have identical dimensions for matrix subtraction.\n");
        exit(EXIT_FAILURE);
    }

    Matrix result = alloc_mat(a.rows, a.cols);

    for (unsigned int i = 0; i < a.rows; i++) {
        for (unsigned int j = 0; j < a.cols; j++) {
            float diff = unchecked_index(&a, i, j) - unchecked_index(&b, i, j);
            unchecked_set_value(&result, i, j, diff);
        }
    }

    return result;
}

void mat_sub_mut(Matrix *a, Matrix b) {
    if (a->rows != b.rows || a->cols != b.cols) {
        printf("Matrix a dimensions: %u x %u\n", a->rows, a->cols);
        printf("Matrix b dimensions: %u x %u\n", b.rows, b.cols);
        fprintf(stderr, "Error: Both matrices must have identical dimensions for in-place matrix subtraction.\n");
        exit(EXIT_FAILURE);
    }

    for (unsigned int i = 0; i < a->rows; i++) {
        for (unsigned int j = 0; j < a->cols; j++) {
            float diff = unchecked_index(a, i, j) - unchecked_index(&b, i, j);
            unchecked_set_value(a, i, j, diff);
        }
    }
}

void scale_mat(Matrix *m, float scalar) {
    for (unsigned int i = 0; i < m->rows; i++) {
        for (unsigned int j = 0; j < m->cols; j++) {
            unchecked_set_value(m, i, j, unchecked_index(m, i, j) * scalar);
        }
    }
}

// Activation functions

float relu(float x) {
    return x <= 0 ? 0 : x;
}

float relu_prime(float x) {
    return x <= 0 ? 0 : 1;
}

Matrix elementwise_relu(Matrix vec) {
    if (vec.cols != 1) {
        fprintf(stderr, "Error: Input must be an nx1 vector for element-wise ReLU.\n");
        exit(EXIT_FAILURE);
    }

    Matrix result = alloc_mat(vec.rows, 1);

    for (unsigned int i = 0; i < vec.rows; i++) {
        float value = unchecked_index(&vec, i, 0);
        float relu_value = relu(value);
        unchecked_set_value(&result, i, 0, relu_value);
    }

    return result;
}

Matrix elementwise_relu_prime(Matrix vec) {
    if (vec.cols != 1) {
        fprintf(stderr, "Error: Input must be an nx1 vector for element-wise ReLU prime.\n");
        exit(EXIT_FAILURE);
    }

    Matrix result = alloc_mat(vec.rows, 1);

    for (unsigned int i = 0; i < vec.rows; i++) {
        float value = unchecked_index(&vec, i, 0);
        float relu_prime_value = relu_prime(value);
        unchecked_set_value(&result, i, 0, relu_prime_value);
    }

    return result;
}

// Cost function
float mse(Matrix predicted, Matrix actual) {
    if (predicted.rows != actual.rows || predicted.cols != actual.cols || predicted.cols != 1) {
        fprintf(stderr, "Error: Vector dimensions must match for mean squared error calculation.\n");
        exit(EXIT_FAILURE);
    }

    Matrix error = mat_sub(actual, predicted);

    float sum = 0.0;

    for (unsigned int i = 0; i < error.rows; i++) {
        float err_value = unchecked_index(&error, i, 0);
        sum += err_value * err_value;
    }
    
    return sum / 2;
}

// input expected to be a 784x1 matrix
void feed_forward(Matrix input, Matrix w1, Matrix b1, Matrix w2, Matrix b2, Matrix w3, Matrix b3, Matrix *z1, Matrix *z2, Matrix *z3, Matrix *a1, Matrix *a2, Matrix *a3) {
    
    Matrix prod = mat_mul(w1, input);

    *z1 = mat_add(prod, b1);
    free_mat(prod);
    
    *a1 = elementwise_relu(*z1);

    Matrix prod2 = mat_mul(w2, *a1);
    *z2 = mat_add(prod2, b2);
    free_mat(prod2);

    *a2 = elementwise_relu(*z2);

    Matrix prod3 = mat_mul(w3, *a2);
    *z3 = mat_add(prod3, b3);
    free_mat(prod3);

    *a3 = elementwise_relu(*z3);
}

// This is the backpropagation step
Matrix calculate_delta(Matrix w, Matrix d_next, Matrix z) {
    Matrix w_t = transpose(w);
    Matrix backwards_error = mat_mul(w_t, d_next);

    Matrix sigma_prime_z = elementwise_relu_prime(z);
    Matrix delta = hadamard_product(backwards_error, sigma_prime_z);

    free_mat(w_t);
    free_mat(backwards_error);
    free_mat(sigma_prime_z);

    return delta;
}

// Training run for a batch
void train_batch(
    unsigned int start_idx, 
    unsigned int num_examples, 
    Matrix *training_inputs, 
    Matrix *training_outputs, 
    Matrix *w1, 
    Matrix *b1, 
    Matrix *w2, 
    Matrix *b2, 
    Matrix *w3, 
    Matrix *b3
) {
    Matrix batch_d1[BATCH_SIZE];
    Matrix batch_d2[BATCH_SIZE];
    Matrix batch_d3[BATCH_SIZE];

    Matrix batch_a1[BATCH_SIZE];
    Matrix batch_a2[BATCH_SIZE];
    Matrix batch_a3[BATCH_SIZE];


    // Forward pass and backprop 
    for (unsigned int i = 0; i < BATCH_SIZE; i++) {
        if (start_idx + i >= num_examples) {
            break;
        } 

        Matrix training_input = training_inputs[start_idx + i];
        Matrix training_output = training_outputs[start_idx + i];

        Matrix z1, a1, z2, a2, z3, a3;
        
        feed_forward(training_input, *w1, *b1, *w2, *b2, *w3, *b3, &z1,&z2, &z3, &a1, &a2, &a3);

        // Calculate loss every 1000 training examples
        if ((start_idx + i + 1) % 1000 == 0) {
            float loss = mse(a3, training_output);
            printf("Loss after %u examples: %.4f\n", start_idx + i + 1, loss);
        }

        // Calculating deltas for final layer
        Matrix cost_pd = mat_sub(a3, training_output); // Vector of partial derivatives of cost function wrt to last layer
        Matrix sigma_prime_z = elementwise_relu_prime(z3); // Vector of derivatives of ReLU for each of the weighted inputs to each neuron in last layer
        Matrix d3 = hadamard_product(cost_pd, sigma_prime_z);
        
        free_mat(cost_pd);
        free_mat(sigma_prime_z);

        // Propagating the deltas backwards
        Matrix d2 = calculate_delta(*w3, d3, z2);
        Matrix d1 = calculate_delta(*w2, d2, z1);

        // Adding deltas and activations to batch arrays
        batch_a1[i] = a1;
        batch_a2[i] = a2;
        batch_a3[i] = a3;
        
        batch_d1[i] = d1;
        batch_d2[i] = d2;
        batch_d3[i] = d3;
    }

    // Calculating batch-average of partial derivatives for each set of weights
    Matrix a0_t = transpose(training_inputs[start_idx]);
    Matrix w_pds1 = mat_mul(batch_d1[0], a0_t);
    free_mat(a0_t);

    Matrix a1_t = transpose(batch_a1[0]);
    Matrix w_pds2 = mat_mul(batch_d2[0], a1_t);
    free_mat(a1_t);

    Matrix a2_t = transpose(batch_a2[0]);
    Matrix w_pds3 = mat_mul(batch_d3[0], a2_t);
    free_mat(a2_t);
    
    // Updating weights
    for (unsigned int i = 1; i < BATCH_SIZE; i++) {
        if (start_idx + i >= num_examples) {
            break;
        }

        if (start_idx + i >= 60000) {
            printf("start_idx + i (%u) is greater than 60000\n", start_idx + i);
            break;
        }

        Matrix a0_t = transpose(training_inputs[start_idx + i]);
        Matrix pd1 = mat_mul(batch_d1[i], a0_t);
        mat_add_mut(&w_pds1, pd1);
        free_mat(pd1);
        free_mat(a0_t);

        Matrix a1_t = transpose(batch_a1[i]);
        Matrix pd2 = mat_mul(batch_d2[i], a1_t);
        mat_add_mut(&w_pds2, pd2);
        free_mat(pd2);
        free_mat(a1_t);

        Matrix a2_t = transpose(batch_a2[i]);
        Matrix pd3 = mat_mul(batch_d3[i], a2_t);
        mat_add_mut(&w_pds3, pd3);
        free_mat(pd3);
        free_mat(a2_t);
    }

    float gradient_scalar = LEARNING_RATE / BATCH_SIZE;

    scale_mat(&w_pds1, gradient_scalar);
    scale_mat(&w_pds2, gradient_scalar);
    scale_mat(&w_pds3, gradient_scalar);

    mat_sub_mut(w1, w_pds1);
    mat_sub_mut(w2, w_pds2); 
    mat_sub_mut(w3, w_pds3);

    free_mat(w_pds1);
    free_mat(w_pds2);
    free_mat(w_pds3);

    // Updating biases
    Matrix b_pds1 = batch_d1[0];
    Matrix b_pds2 = batch_d2[0];
    Matrix b_pds3 = batch_d3[0];

    for (unsigned int i = 1; i < BATCH_SIZE; i++) {
        if (start_idx + i >= num_examples) {
            break;
        } 

        mat_add_mut(&b_pds1, batch_d1[i]);
        mat_add_mut(&b_pds2, batch_d2[i]);
        mat_add_mut(&b_pds3, batch_d3[i]);
    }

    scale_mat(&b_pds1, gradient_scalar);
    scale_mat(&b_pds2, gradient_scalar);
    scale_mat(&b_pds3, gradient_scalar);

    mat_sub_mut(b1, b_pds1);
    mat_sub_mut(b2, b_pds2);
    mat_sub_mut(b3, b_pds3);

    // Free batch_d1, batch_d2, batch_d3, batch_a1, batch_a2, batch_a3
    for (unsigned int i = 0; i < BATCH_SIZE; i++) {
        if (start_idx + i >= num_examples) {
            break;
        } 

        free_mat(batch_d1[i]);
        free_mat(batch_d2[i]);
        free_mat(batch_d3[i]);
        free_mat(batch_a1[i]);
        free_mat(batch_a2[i]);
        free_mat(batch_a3[i]);
    }
}

// File IO
void load_mnist_data(char *filename, Matrix **outputs, Matrix **inputs, unsigned int *num_examples) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error: Failed to open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Count the number of examples in the file
    unsigned int count = 0;
    char buffer[2048];
    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        count++;
    }

    printf("Number of examples: %u\n", count);
    rewind(file);

    *num_examples = count;
    *outputs = (Matrix *)malloc(count * sizeof(Matrix));
    *inputs = (Matrix *)malloc(count * sizeof(Matrix));

    for (unsigned int i = 0; i < count; i++) {
        int label;
        char pixel_str[1024];
        
        if (fscanf(file, "%d %s", &label, pixel_str) != 2) {
            printf("Invalid data format on line %d\n", i + 1);
            break;
        }

        // Create output matrix
        Matrix output = alloc_mat(10, 1);
        for (unsigned int j = 0; j < 10; j++) {
            unchecked_set_value(&output, j, 0, (j == label) ? 1.0 : 0.0);
        }

        (*outputs)[i] = output;

        // Create input matrix
        Matrix input = alloc_mat(IN_LAYER_LEN, 1);
        char *token = strtok(pixel_str, ",");
        unsigned int idx = 0;
        while (token != NULL) {
            float pixel_value = (float)atoi(token) / 255.0;
            unchecked_set_value(&input, idx, 0, pixel_value);
            idx++;
            token = strtok(NULL, ",");
        }

        (*inputs)[i] = input;

    }

    fclose(file);
}

float test_model(Matrix w1, Matrix b1, Matrix w2, Matrix b2, Matrix w3, Matrix b3) {
    unsigned int num_examples;
    Matrix *outputs, *inputs;
    load_mnist_data("mnist_test.txt", &outputs, &inputs, &num_examples);

    num_examples = 10000;
    unsigned int correct_guesses = 0;
    for (unsigned int i = 0; i < num_examples; i++) {
        Matrix input = inputs[i];
        Matrix expected_output = outputs[i];

        Matrix z1, a1, z2, a2, z3, a3;

        feed_forward(input, w1, b1, w2, b2, w3, b3, &z1, &z2, &z3, &a1, &a2, &a3);

        unsigned int predicted_label = 0;
        float max_activation = unchecked_index(&a3, 0, 0);
        for (unsigned int j = 1; j < a3.rows; j++) {
            float activation = unchecked_index(&a3, j, 0);
            if (activation > max_activation) {
                max_activation = activation;
                predicted_label = j;
            }
        }

        unsigned int expected_label = 0;
        float expected_value = unchecked_index(&expected_output, 0, 0);
        for (unsigned int j = 1; j < expected_output.rows; j++) {
            float value = unchecked_index(&expected_output, j, 0);
            if (value > expected_value) {
                expected_value = value;
                expected_label = j;
            }
        }

        if (predicted_label == expected_label) {
            correct_guesses++;
        }

        free_mat(z1);
        free_mat(a1);
        free_mat(z2);
        free_mat(a2);
        free_mat(z3);
        free_mat(a3);
    }

    for (unsigned int i = 0; i < num_examples; i++) {
        free_mat(inputs[i]);
        free_mat(outputs[i]);
    }
    free(inputs);
    free(outputs);

    return (float)correct_guesses / num_examples;
}


int main() {

    Matrix w1 = rand_matrix(H1_LEN, IN_LAYER_LEN);
    Matrix b1 = zero_vector(H1_LEN);

    Matrix w2 = rand_matrix(H2_LEN, H1_LEN);
    Matrix b2 = zero_vector(H2_LEN);

    Matrix w3 = rand_matrix(OUT_LAYER_LEN, H2_LEN);
    Matrix b3 = zero_vector(OUT_LAYER_LEN);


    Matrix *train_inputs, *train_outputs;
    unsigned int num_train_examples;
    load_mnist_data("mnist_train.txt", &train_outputs, &train_inputs, &num_train_examples);

    unsigned int start_idx = 0;
    num_train_examples = 60000;
    for (unsigned int epoch = 0; epoch < EPOCHS; epoch++) {

        while (start_idx < num_train_examples) {
            train_batch(start_idx, num_train_examples, train_inputs, train_outputs, &w1, &b1, &w2, &b2, &w3, &b3);
            start_idx += BATCH_SIZE;
        }
        start_idx = 0;
    }

    // Test the model
    float accuracy = test_model(w1, b1, w2, b2, w3, b3);
    printf("Test accuracy: %.2f%%\n", accuracy * 100);

    return 0;
}
