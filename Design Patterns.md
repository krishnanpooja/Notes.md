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

