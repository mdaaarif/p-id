with open('line_tag_merge_4.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Make sure we carry Line_Style to the final output CSV
code = code.replace('''"From_Equip_Class",
            "To_Equip_Class"
        ]''', '''"From_Equip_Class",
            "To_Equip_Class",
            "Line_Style"
        ]''')

code = code.replace('''"Distance_Pixels":          round(distance, 2)  if matched else None,
                "Confidence_Score":         self._compute_confidence(
                    distance if matched else None
                )
            })''', '''"Distance_Pixels":          round(distance, 2)  if matched else None,
                "Confidence_Score":         self._compute_confidence(
                    distance if matched else None
                ),
                "Line_Style":               group_df["Line_Style"].iloc[0] if "Line_Style" in group_df.columns else "Solid"
            })''')

with open('line_tag_merge_4.py', 'w', encoding='utf-8') as f:
    f.write(code)
