
MEd = 480
beff=450
bw=300
h=450
hf=150
fi=8
fck=25
fyk=500
fcd=fck/1.4
fyd=fyk/1.15
a1=42
d=h-a1
a2=42
xeff=201.3425287



As2 = (-xeff * bw * fcd * (d - 0.5 * xeff) - hf * (beff - bw) * fcd * (d - 0.5 * hf) + MEd * 1e6) / (fyd * (d - a2))
print(As2)