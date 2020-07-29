**Singleton Class in Java**

In object-oriented programming, a singleton class is a class that can have only one object (an instance of the class) at a time.
After first time, if we try to instantiate the Singleton class, the new variable also points to the first instance created. So whatever modifications we do to any variable inside the class through any instance, it affects the variable of the single instance created and is visible if we access that variable through any variable of that class type defined.
To design a singleton class:

Make constructor as private.
Write a static method that has return type object of this singleton class. Here, the concept of Lazy initialization is used to write this static method.

```
// Java program implementing Singleton class 
// with getInstance() method 
class Singleton 
{ 
    // static variable single_instance of type Singleton 
    private static Singleton single_instance = null; 
  
    // variable of type String 
    public String s; 
  
    // private constructor restricted to this class itself 
    private Singleton() 
    { 
        s = "Hello I am a string part of Singleton class"; 
    } 
  
    // static method to create instance of Singleton class 
    public static Singleton getInstance() 
    { 
        if (single_instance == null) 
            single_instance = new Singleton(); 
  
        return single_instance; 
    } 
} 
  
// Driver Class 
class Main 
{ 
    public static void main(String args[]) 
    { 
        // instantiating Singleton class with variable x 
        Singleton x = Singleton.getInstance();
   }
}
```

**Factory Pattern**

 factory pattern is one of the core design principles to create an object, allowing clients to create objects of a library(explained below) in a way such that it doesn’t have tight coupling with the class hierarchy of the library.
 
 ```
 Vehicle* Vehicle::Create(VehicleType type) { 
    if (type == VT_TwoWheeler) 
        return new TwoWheeler(); 
    else if (type == VT_ThreeWheeler) 
        return new ThreeWheeler(); 
    else if (type == VT_FourWheeler) 
        return new FourWheeler(); 
    else return NULL; 
} 
```

**Adapter Pattern**

To use an adapter:

The client makes a request to the adapter by calling a method on it using the target interface.
The adapter translates that request on the adaptee using the adaptee interface.
Client receive the results of the call and is unaware of adapter’s presence.
// Java implementation of Adapter pattern 

interface Bird 
{ 
	// birds implement Bird interface that allows 
	// them to fly and make sounds adaptee interface 
	public void fly(); 
	public void makeSound(); 
} 

class Sparrow implements Bird 
{ 
	// a concrete implementation of bird 
	public void fly() 
	{ 
		System.out.println("Flying"); 
	} 
	public void makeSound() 
	{ 
		System.out.println("Chirp Chirp"); 
	} 
} 

interface ToyDuck 
{ 
	// target interface 
	// toyducks dont fly they just make 
	// squeaking sound 
	public void squeak(); 
} 

class PlasticToyDuck implements ToyDuck 
{ 
	public void squeak() 
	{ 
		System.out.println("Squeak"); 
	} 
} 

class BirdAdapter implements ToyDuck 
{ 
	// You need to implement the interface your 
	// client expects to use. 
	Bird bird; 
	public BirdAdapter(Bird bird) 
	{ 
		// we need reference to the object we 
		// are adapting 
		this.bird = bird; 
	} 

	public void squeak() 
	{ 
		// translate the methods appropriately 
		bird.makeSound(); 
	} 
} 

class Main 
{ 
	public static void main(String args[]) 
	{ 
		Sparrow sparrow = new Sparrow(); 
		ToyDuck toyDuck = new PlasticToyDuck(); 

		// Wrap a bird in a birdAdapter so that it 
		// behaves like toy duck 
		ToyDuck birdAdapter = new BirdAdapter(sparrow); 

		System.out.println("Sparrow..."); 
		sparrow.fly(); 
		sparrow.makeSound(); 

		System.out.println("ToyDuck..."); 
		toyDuck.squeak(); 

		// toy duck behaving like a bird 
		System.out.println("BirdAdapter..."); 
		birdAdapter.squeak(); 
	} 
} 


