from PIL import Image, ImageDraw, ImageFont



#draw.line((0, im.height, im.width, 0), fill=(255, 0, 0), width=8)
#draw.rectangle((100, 100, 200, 200), fill=(0, 255, 0))

# for i in range(7):
#     x1 = 0
#     y1 = 0
#     x2 = 2*4*(i+1)
#     y2 = 2*4*(i+1)
#     for j in range((64-2*4*(i+1))*(64-2*4*(i+1))):
#         im = Image.new("L", (64, 64), (0))
#         draw = ImageDraw.Draw(im)
#         draw.ellipse((x1, y1, x2, y2), fill=(255))
#         im.save('ellipse/ellipse_d'+ str((i+1)) +'_p' + str((j+1)) + '.jpg', quality=95)
#         if x2==63:
#             x1 = 0
#             x2 = 2*4*(i+1)
#             y1 += 1
#             y2 += 1
#         else:
#             x1 += 1
#             x2 += 1
        


for i in range(7):
    x1 = 0
    y1 = 0
    x2 = 2*4*(i+1)
    y2 = 2*4*(i+1)
    for j in range((64-2*4*(i+1))*(64-2*4*(i+1))):
        im = Image.new("L", (64, 64), (0))
        draw = ImageDraw.Draw(im)
        draw.rectangle((x1, y1, x2, y2), fill=(255))
        if x2==63:
            x1 = 0
            x2 = 2*4*(i+1)
            y1 += 1
            y2 += 1
        else:
            x1 += 1
            x2 += 1
        im.save('square/square_d'+ str((i+1)) +'_p' + str((j+1)) + '.jpg', quality=95)



for i in range(7):
    x1 = 4*(i+1)
    y1 = 0
    x2 = 0
    y2 = 8*(i+1)
    x3 = 8*(i+1)
    y3 = 8*(i+1)
    for j in range((64-2*4*(i+1))*(64-2*4*(i+1))):
        im = Image.new("L", (64, 64), (0))
        draw = ImageDraw.Draw(im)
        draw.polygon(((x1, y1), (x2, y2), (x3, y3)), fill=(255))
        if x3==63:
            x1 = 4*(i+1)
            x2 = 0
            x3 = 8*(i+1)
            y1 += 1
            y2 += 1
            y3 += 1
        else:
            x1 += 1
            x2 += 1
            x3 += 1
        im.save('triangle/triangle_d'+ str((i+1)) +'_p' + str((j+1)) + '.jpg', quality=95)
