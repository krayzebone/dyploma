from predict_section.predict_rect_section import predict_rect_section_properties

# Example usage
results = predict_rect_section_properties(
    MEd=157.163915,
    b=1306.223854,
    h=629.038043,
    d=575.038043,
    fi=28,
    fck=35,
    ro1=0.016487,
    ro2=0.010492
)

print(results)
