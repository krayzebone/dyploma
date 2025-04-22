from section_calcs.T_Section_1r import check_section_type_1r, PT_Section_1r, RZT_Section_1r
from section_calcs.T_Section_2r import check_section_type_2r, PT_Section_2r, RZT_Section_2r
from section_calcs.T_Section_3r import check_section_type_3r, PT_Section_3r, RZT_Section_3r

MEd = 220
beff = 400
bw = 200
h = 400
hf = 150
fck = 25
figl = 32
fyk = 500
r = 3

if r == 1:
    section_type = check_section_type_1r(MEd, beff, h, hf, figl, fck)

    if section_type == 'PT':
        section_type, As1, As2 = PT_Section_1r(MEd, beff, h, figl, fck)

    if section_type == 'RZT':
        section_type, As1, As2 = RZT_Section_1r(MEd, beff, bw, h, hf, figl, fck)

elif r == 2:
    section_type = check_section_type_2r(MEd, beff, h, hf, figl, fck)

    if section_type == 'PT':
        section_type, As1, As2 = PT_Section_2r(MEd, beff, h, figl, fck)

    if section_type == 'RZT':
        section_type, As1, As2 = RZT_Section_2r(MEd, beff, bw, h, hf, figl, fck)

elif r == 3:
    section_type = check_section_type_3r(MEd, beff, h, hf, figl, fck)

    if section_type == 'PT':
        section_type, As1, As2 = PT_Section_3r(MEd, beff, h, figl, fck)

    if section_type == 'RZT':
        section_type, As1, As2 = RZT_Section_3r(MEd, beff, bw, h, hf, figl, fck)

else:
    None

print(section_type)
print(As1)
print(As2)
