import math

MEqp = 80.955 * 1e6

h = 500
hf = 150
beff = 1636
bw = 300
cnom = 30
fistr = 8

n1 = 3
n2 = 0
fi = 25

As1 = n1 * fi**2 * math.pi / 4
As2 = n2 * fi**2 * math.pi / 4

Es = 200_000
fck = 25
Ecm = 32837 #####

Eceff = 10798

fctm = 2.9 #######

#a1 = cnom + fi / 2 + fistr
a1 = 48
a2 = cnom + fi / 2 + fistr
d = h - a1

alpha_cs = Es / Eceff

# Pole przekroju betonu
Ac = 0.305 * 1e6

# Obwód części przekroju wystawionej na wysychanie
u = beff + 2*(h - hf)

# Miarodajny wymair przekroju
h0 = 2 * (Ac / u)

# Pole sprowadzonego pola przekroju
Acs = Ac + alpha_cs * (As1 + As2)

# Moment statyczny przekroju
Scs = beff * hf * (h - 0.5 * hf) + (h - hf) * bw * 0.5 * (h - hf)

# Odległość osi centralnej przekroju od osi przyjętej
xI = Scs / Acs

#  Moment bezwładności przekroju w fazie I
II = (beff - bw) * hf**3 / 12 + (beff - bw) * hf * (h - xI - 0.5 * hf)**2 + (bw * h**3 / 12) + bw * h * (0.5 * h - xI)**2
+ alpha_cs*(As1*(d - xI)**2 + As2*(xI - a1)**2)

# Wskaźnik na zginanie przekroju
Wcs = II / (h - xI)

# Moment krytyczny
Mcr = fctm * Wcs

######
# II faza
######

xII_1 = -(math.sqrt(
        alpha_cs * (2 * beff * a2 * As2 + 2 * beff * d * As1 + alpha_cs * As2**2 + 2 * alpha_cs * As2 * As1 + alpha_cs * As1**2))
        + alpha_cs * As2 + alpha_cs * As1) / beff

xII_2 = (math.sqrt(
        alpha_cs * (2 * beff * a2 * As2 + 2 * beff * d * As1 + alpha_cs * As2**2 + 2 * alpha_cs * As2 * As1 + alpha_cs * As1**2))
        - alpha_cs * As2 - alpha_cs * As1) / beff

print (xII_1, xII_2)

if xII_1 < 0:
    xII = xII_2
else:
    xII = xII_1

III = beff * xII**3 / 3 + alpha_cs*(As1*(d - xII)**2 + As2*(xII - a1)**2)


sigmas = alpha_cs * MEqp / III * (d - xII)
kt = 0.4
fctm = 2.565