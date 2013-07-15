kernel void square(global float *in, global float *out)
{
    size_t i = get_global_id(0);
    out[i] = in[i] * in[i];
}