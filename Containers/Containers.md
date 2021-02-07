## Why Containers?

- license, trust, install, cross-platform, conflicting lib, port conflicts,uninstall ??
- Distribution. Installation, Running/operations 
- Images help with distribution (Download, license, package, trust)
- Containers help with installation, operations
- process to start and stop different technology will be the same

## What is Container?
- Containers are processes
- Will be listed in `ps` and task manager(windows)
- They are isolated processes - this helps avoid conflicts
- Virtual memory isolation - processes are loaded into their Virtual address space. Contents that don't fit into physical memory- swapped into hard drive temp.
Helps use more memory than the system actually has....
- snapshots are maintained when we switch from one process to another

### Opt-in isolation?
- Resources cannot be accessed outside the containers
- common networking between containers is possible
- `chroot`- change root directory - process sees a different root directory
- `capabilities` - provide capabilty to bind to a particular port , bypass and rwx a particular file

### Mount Namespaces 
- File systems are mounted to a particular space
- Isolating the storage
- cgroups -Control grps
  - cgroups is a linux kernel that limits, accounts for and isolated the resource usage to a collection of proceesses
- easier way to remember, cgroup is like giving a slice of pizza to each team(process) and namespace is a like giving a pizza to each group

### Namespaces
- Different types of namespaces - Cgroup, IPC, Mount, UTS
- Mount - two processes are on different Mount namespaces - thereby, different filesystems
- UTS namespace - different processes have different hostname vag@vag, though the processes are running on the same system
- IPC -inter process communication -  `ipcs` ->lists the resources for the processes 

- 

