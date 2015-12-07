#include <stdio.h> 
#include <stdlib.h>

typedef struct
{
    float fx;
    float fy;
    float mag2;
} value_t;

static int comp(const void* elem1, const void* elem2)
{
    value_t* v1 = (value_t*)elem1;
    value_t* v2 = (value_t*)elem2;

    if(v1->mag2 > v2->mag2) return  1;
    if(v1->mag2 < v2->mag2) return -1;

    return 0;
}

void c_downsample (float* xyflow, float* down, int m, int n, int f) 
{
	//printf("c_downsample %d %d %d\n", m, n, f); 

    int dm = m / f;
    int dn = n / f;

    int f2 = f * f;
    value_t* buffer = malloc(sizeof(value_t) * f2);

    int di;
    int dj;

    for (di = 0; di < dm; di++) //height rows
        for (dj = 0; dj < dn; dj++) //width cols
        {
            int i = di * f;
            int j = dj * f;

            value_t* ptr = buffer;
            int l;
            int k;
            for(l=0; l<f; l++)
                for(k=0; k<f; k++)
                {
                    int idx = (n*(i+l) + (j+k)) * 2;
                    ptr->fx = xyflow[idx+0];
                    ptr->fy = xyflow[idx+1];
                    ptr->mag2 = ptr->fx*ptr->fx + ptr->fy*ptr->fy;
                    ptr++;
                }

            qsort(buffer, f2, sizeof(value_t), comp);

            int didx = (dn*di + dj) * 2;
            down[didx+0] = buffer[f2/2].fx / f;
            down[didx+1] = buffer[f2/2].fy / f;
		}

    free(buffer);
}
