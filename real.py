from defisheye import Defisheye


dtype = 'stereographic'
format = 'fullframe'
fov = 140
pfov = 100

img = "images\\04_11\\Image__2022-11-04__09-10-24_11zon.jpg"
img_out = f"images\\04_11\\out\\result_{dtype}_{format}_{pfov}_{fov}.jpg"

obj = Defisheye(img, dtype=dtype, format=format, fov=fov, pfov=pfov)
obj.convert(img_out)

print("done")
