#include<cstdio>
#include<cstdlib>

static void compute(float *output, float *input0, float *input1, int num) {
  for (int ii = 0; ii < num; ++ii) {
    output[ii] = input0[ii] + input1[ii];
  }
}

static float *generate_data(int n) {
  float *data = (float *)malloc(sizeof(float) * n);
  for (int i = 0; i < n; i++) {
    data[i] = ((float)(rand() % 20) - 5) / 5.0f;
  }
  return data;
}

static void write_data(const char *file_name, float *data, int num) {
  FILE *handle = fopen(file_name, "w");
  fprintf(handle, "%d", num);
  for (int ii = 0; ii < num; ii++) {
    fprintf(handle, "\n%.2f", *data++);
  }
  fflush(handle);
  fclose(handle);
}

static void create_dataset(int dim) {
	
  const char *input0_file_name = "input0.raw";
  const char *input1_file_name = "input1.raw";
  const char *output_file_name = "expected.raw";

  float *input0_data = generate_data(dim);
  float *input1_data = generate_data(dim);
  float *output_data = (float *)calloc(sizeof(float), dim);

  compute(output_data, input0_data, input1_data, dim);

  write_data(input0_file_name, input0_data, dim);
  write_data(input1_file_name, input1_data, dim);
  write_data(output_file_name, output_data, dim);

  free(input0_data);
  free(input1_data);
  free(output_data);
}

int main() {

  create_dataset(35);
  return 0;
}