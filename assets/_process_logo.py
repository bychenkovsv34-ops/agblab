# -*- coding: utf-8 -*-
"""Отделяем красный знак AGBLAB AI от тёмного фона -> прозрачный PNG."""
from PIL import Image
import os

SRC = r"C:\Users\Sergey\Pictures\ВИЗУАЛ ОТ ИИ\ChatGPT Image 5 июл. 2025 г., 11_33_26.png"
OUT = os.path.dirname(__file__)

im = Image.open(SRC).convert("RGBA")
print("source size:", im.size)
px = im.load()
w, h = im.size

# alpha по "красности": R заметно больше G и B  ->  логотип; иначе фон/тень
LO, HI = 18, 70  # порог red-dominance
for y in range(h):
    for x in range(w):
        r, g, b, _ = px[x, y]
        red = r - max(g, b)
        if red <= LO:
            a = 0
        elif red >= HI:
            a = 255
        else:
            a = int((red - LO) / (HI - LO) * 255)
        # лёгкое усиление яркости красного, чтобы знак был сочнее
        px[x, y] = (r, g, b, a)

# обрезаем прозрачные поля
bbox = im.getbbox()
print("bbox:", bbox)
im = im.crop(bbox)
W, H = im.size
print("trimmed:", im.size)

# полный логотип (знак + надпись)
logo_path = os.path.join(OUT, "logo.png")
im.save(logo_path)
print("saved", logo_path)

# бейдж — только шестиугольник: ищем прозрачный зазор после первого блока
bpx = im.load()
cols = [any(bpx[x, y][3] > 12 for y in range(H)) for x in range(W)]
seen = False
gap = W
for x in range(W):
    if cols[x]:
        seen = True
    elif seen:
        gap = x
        break
badge = im.crop((0, 0, gap, H))
print("badge width:", gap)
badge_path = os.path.join(OUT, "badge.png")
badge.save(badge_path)
print("saved", badge_path)

# favicon 64x64 из бейджа на прозрачном квадрате
fav = badge.copy()
side = max(fav.size)
canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
canvas.paste(fav, ((side - fav.size[0]) // 2, (side - fav.size[1]) // 2), fav)
canvas = canvas.resize((64, 64), Image.LANCZOS)
fav_path = os.path.join(OUT, "favicon.png")
canvas.save(fav_path)
print("saved", fav_path)
