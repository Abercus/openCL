
void apply_force_gpu(__global particle_t &particle, __global particle_t &neighbor)
{
  double dx = neighbor.x - particle.x;
  double dy = neighbor.y - particle.y;
  double r2 = dx * dx + dy * dy;
  if( r2 > cutoff*cutoff )
      return;
  r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
  double r = sqrt( r2 );

  //
  //  very simple short-range repulsive force
  //
  double coef = ( 1 - cutoff / r ) / r2 / mass;
  particle.ax += coef * dx;
  particle.ay += coef * dy;

}


// How do we apply forces is the question.
__kernel void compute_forces_gpu(__global particle_t * particles, __global int n)
{
  // Get thread (particle) ID
 // int tid = threadIdx.x + blockIdx.x * blockDim.x;
  //  int tid = get_local_id(0) + get_group_id(0) * get_local_size(0);
	int tid = get_global_id(0);
  if(tid >= n) return;

  particles[tid].ax = particles[tid].ay = 0;
  for(int j = 0 ; j < n ; j++)
    apply_force_gpu(particles[tid], particles[j]);

}

__kernel void move_gpu (__global particle_t * particles, __global int n, __global double size)
{

  // Get thread (particle) ID
 // int tid = threadIdx.x + blockIdx.x * blockDim.x;
  
  //int tid = get_local_id(0) + get_group_id(0) * get_local_size(0);
  	int tid = get_global_id(0);

  if(tid >= n) return;

  particle_t * p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x  += p->vx * dt;
    p->y  += p->vy * dt;

    //
    //  bounce from walls
    //
    while( p->x < 0 || p->x > size )
    {
        p->x  = p->x < 0 ? -(p->x) : 2*size-p->x;
        p->vx = -(p->vx);
    }
    while( p->y < 0 || p->y > size )
    {
        p->y  = p->y < 0 ? -(p->y) : 2*size-p->y;
        p->vy = -(p->vy);
    }

}