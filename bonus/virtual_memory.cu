#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ void init_invert_page_table(VirtualMemory *vm) {

  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->invert_page_table[i] = 0x80000000; 
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = i;
    vm->invert_page_table[i + 2 * vm->PAGE_ENTRIES] = 0;
  }
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES) {
  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;

  // init constants
  vm->PAGESIZE = PAGESIZE;                  /* page size = 32b */
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;    /* 16KB for page table setting */
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;  /* 32KB for physical memory */
  vm->STORAGE_SIZE = STORAGE_SIZE;          /* 128KB for the disk storage */
  vm->PAGE_ENTRIES = PAGE_ENTRIES;          /* = PHYSICAL_MEM_SIZE / PAGE_SIZE =  1024*/

  // before first vm_write or vm_read
  init_invert_page_table(vm);
}

__device__ int find_frame_number(VirtualMemory *vm, int target) {
  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    if (target == vm->invert_page_table[i + vm->PAGE_ENTRIES]&& vm->invert_page_table[i] != 0x80000000) {
      for (int j = 0; j < vm->PAGE_ENTRIES; j++){
        vm->invert_page_table[j + 2 * vm->PAGE_ENTRIES] += 1;
      }
      return i;
    }
  }
  return -1;
}

__device__ int get_LRU_position(VirtualMemory *vm) {
  // empty
  int max = vm->invert_page_table[0 + 2 * vm->PAGE_ENTRIES];
  int res = 0;
  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    if (vm->invert_page_table[i + 2 * vm->PAGE_ENTRIES] > max)  {
      max = vm->invert_page_table[i + 2 * vm->PAGE_ENTRIES];
      res = i;
    }
  }
  return res;
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {

  __syncthreads();
	if (addr % 4 != ((int)threadIdx.x)) return;
  /* Complete vm_write function to write value into data buffer */
  int pid_vpn = addr / vm->PAGESIZE;      
  int offset = addr % vm->PAGESIZE;          
  int frame_number = find_frame_number(vm, pid_vpn);
  uchar res;

  if (frame_number != -1){ // exist
    vm->invert_page_table[frame_number + 2 * vm->PAGE_ENTRIES] = 0;
    res = vm->buffer[frame_number*vm->PAGESIZE+offset];           
  }

  else{ // not exist
    *vm->pagefault_num_ptr += 1;

    int empty = -1; // to find if there exist empty idx
    for (int i = 0; i < vm->PAGE_ENTRIES; i++){  
      if (vm->invert_page_table[i] == 0x80000000){ 
          empty = i;               
          break;
      }
    }

    if (empty != -1) { // empty
      for (int i = 0; i < vm->PAGESIZE; i++){
        vm->buffer[empty*vm->PAGESIZE+i] = vm->storage[pid_vpn*vm->PAGESIZE+i]; // swap in
      }
      // update page table
      vm->invert_page_table[empty+vm->PAGE_ENTRIES] = pid_vpn;
      vm->invert_page_table[empty] = 0; 
      vm->invert_page_table[empty + 2 * vm->PAGE_ENTRIES] = 0; // update lru table
      res = vm->buffer[empty*vm->PAGESIZE+offset];
    }

    else { // no empty, so find lru
      int LRU_idx = get_LRU_position(vm); // victim
      int LRU_disk_idx = vm->invert_page_table[LRU_idx + vm->PAGE_ENTRIES];
      for (int i = 0; i < vm->PAGESIZE; i++){                                            
        vm->storage[LRU_disk_idx*vm->PAGESIZE+i] = vm->buffer[LRU_idx*vm->PAGESIZE+i];  // swap out
        vm->buffer[LRU_idx*vm->PAGESIZE+i] = vm->storage[pid_vpn*vm->PAGESIZE+i]; // swap in
      }
      // update page table
      vm->invert_page_table[LRU_idx+vm->PAGE_ENTRIES] = pid_vpn;
      vm->invert_page_table[LRU_idx] = 0; 
      vm->invert_page_table[LRU_idx + 2 * vm->PAGE_ENTRIES] = 0; // update lru table
      res = vm->buffer[LRU_idx*vm->PAGESIZE+offset];
    }

  }

  return res;
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  __syncthreads();
	if (addr % 4 != ((int)threadIdx.x)) return;
  /* Complete vm_write function to write value into data buffer */
  int pid_vpn = addr / vm->PAGESIZE;      
  int offset = addr % vm->PAGESIZE;           /* get the last 5 bits */
  int frame_number = find_frame_number(vm, pid_vpn);

  if (frame_number != -1) {
    vm->invert_page_table[frame_number + 2 * vm->PAGE_ENTRIES] = 0;
    vm->buffer[frame_number*vm->PAGESIZE+offset] = value;
  }

  else{ 
    *vm->pagefault_num_ptr += 1;

    int empty = -1;
    for (int i = 0; i < vm->PAGE_ENTRIES; i++){  
      if (vm->invert_page_table[i] == 0x80000000){ 
          empty = i;               
          break;
      }
    }

    if (empty != -1) {
      for (int i = 0; i < vm->PAGESIZE; i++){
        vm->buffer[empty*vm->PAGESIZE+i] = vm->storage[pid_vpn*vm->PAGESIZE+i]; 
      }

      vm->invert_page_table[empty+vm->PAGE_ENTRIES] = pid_vpn;
      vm->invert_page_table[empty] = 0; 
      vm->invert_page_table[empty + 2 * vm->PAGE_ENTRIES] = 0;
      vm->buffer[empty*vm->PAGESIZE+offset] = value;
    }

    else {
      int LRU_idx = get_LRU_position(vm); // victim
      int LRU_disk_idx = vm->invert_page_table[LRU_idx + vm->PAGE_ENTRIES]; // pid_vpn
      for (int i = 0; i < vm->PAGESIZE; i++){                                              
        vm->storage[LRU_disk_idx*vm->PAGESIZE+i] = vm->buffer[LRU_idx*vm->PAGESIZE+i];   
        vm->buffer[LRU_idx*vm->PAGESIZE+i] = vm->storage[pid_vpn*vm->PAGESIZE+i]; 
      }
      vm->invert_page_table[LRU_idx+vm->PAGE_ENTRIES] = pid_vpn;
      vm->invert_page_table[LRU_idx] = 0; 
      vm->invert_page_table[LRU_idx + 2 * vm->PAGE_ENTRIES] = 0; 
      vm->buffer[LRU_idx*vm->PAGESIZE+offset] = value;
    }
  }
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
  for (int i = 0; i < input_size / 4; i++){
    results[i * 4 + (int)threadIdx.x] = vm_read(vm, i * 4 + (int)threadIdx.x+offset);
  }
}