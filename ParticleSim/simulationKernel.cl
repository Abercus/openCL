
#define density 0.0005
#define mass    0.01
#define cutoff  0.01
#define min_r   (cutoff/100)
#define dt      0.0005

//
// particle data structure
//
typedef struct 
{
  double x;
  double y;
  double vx;
  double vy;
  double ax;
  double ay;
} particle_t;

void apply_force_gpu(__global particle_t *particle, __global particle_t *neighbor)
{
  double dx = neighbor->x - particle->x;
  double dy = neighbor->y - particle->y;
  double r2 = dx * dx + dy * dy;
  if( r2 > cutoff*cutoff )
      return;
  r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
  double r = sqrt( r2 );

  //
  //  very simple short-range repulsive force
  //
  double coef = ( 1 - cutoff / r ) / r2 / mass;
  particle->ax += coef * dx;
  particle->ay += coef * dy;

}


// How do we apply forces is the question.
__kernel void compute_forces_gpu(__global particle_t * particles, int n)
{
  // Get thread (particle) ID
 // int tid = threadIdx.x + blockIdx.x * blockDim.x;
  //  int tid = get_local_id(0) + get_group_id(0) * get_local_size(0);
	int tid = get_global_id(0);
  if(tid >= n) return;

  particles[tid].ax = particles[tid].ay = 0;
  for(int j = 0 ; j < n ; j++)
    apply_force_gpu(&particles[tid], &particles[j]);

}

__kernel void move_gpu (__global particle_t * particles, int n, double size)
{

  // Get thread (particle) ID
 // int tid = threadIdx.x + blockIdx.x * blockDim.x;
  
  //int tid = get_local_id(0) + get_group_id(0) * get_local_size(0);
  	int tid = get_global_id(0);

  if(tid >= n) return;

   //particle_t *p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    particles[tid].vx += particles[tid].ax * dt;
    particles[tid].vy += particles[tid].ay * dt;
    particles[tid].x  += particles[tid].vx * dt;
    particles[tid].y  += particles[tid].vy * dt;

    //
    //  bounce from walls
    //
    while( particles[tid].x < 0 || particles[tid].x > size )
    {
        particles[tid].x  = particles[tid].x < 0 ? -(particles[tid].x) : 2*size-particles[tid].x;
        particles[tid].vx = -(particles[tid].vx);
    }
    while( particles[tid].y < 0 || particles[tid].y > size )
    {
        particles[tid].y  = particles[tid].y < 0 ? -(particles[tid].y) : 2*size-particles[tid].y;
        particles[tid].vy = -(particles[tid].vy);
    }

}
