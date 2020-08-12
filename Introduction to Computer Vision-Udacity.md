## Lesson 14- Camera and Images

#### Pinhole cameras:
A small hole allows only the required rays of light to pass. This captures the image in upside down. The size of the hole called the aperture.Ideal size is found to be 0.35mm. Larger the hole the image gets blurry. Smaller than 0.35mm causes diffraction which makes the image blurry again.

#### Thin lens:
Parallel rays meet at the focal point. The distance between the image and object to the lens, is computed using similar triangles.

Thin lens's eqn: 
z': distance of image film from lens
z: distance of the obj from lens
f: focal length
1/z'+z/z=1/f

#### Depth of Field:
Wide open aperture if you want to seperate foregound from background
Landscape aperture required is small. Everything in focus

#### Field of View (FOV)
- How wide is the view?
- Increase focal length of the lens, causes the zooming effect.
- Tripod is required for a stable picture. Less the image is going to shaky. 
- Longer the focal length==smaller field of view.


## Lession 15 - Perspective Imaging

#### Coordinate System
Right handed coordinate system. Assuming img plane is placed such that the img is not inverted. 
Projection Eqn:
Use similar Triangles
(X,Y,Z) -> (-d(X/Z), -d(Y/Z),-d)
Divide by Z, shows the obj farther away is smaller
Zero is the centre of the image
Causes: The moon to move with you, North star always in the North and the Sun's postion telling the time of the day

#### Homo Coordinates
Converting from homo coord:
[x y w]T => (x/w,y/w)

Perspective Projection: 
Multiply the img mat with the homo coordinates. This results the linear representation of the points.

#### Parallel Lines
Almost all Parallel lines in the World meet in the image:-)  Not the vertical ones!!
The point is called Vanishing Point

Too much math to prove:-p

#### Other models:
1. Orthographic  - Distance from center of proj to img plane is infinite
2. Weak Perspective - (x,y,z)-> (fx/z,fy/z).
   Each object has its own scaling. Scale factor per object. 
 
## Lesson 16 - Stereo Geometry
- Img from one eye is different from img for other eye
- 3D is possible bcs two different are shown to the eye.

#### Depth of Disparity
If in an image the object wrt to its background, for example a chimney has moved left wrt to tree in background, then its right stereo image
Disparity Map- conatins disparity for every point (x,y)
Brighter values closer z(which is an estimate of depth)


## Lesson 17 - 




































 

