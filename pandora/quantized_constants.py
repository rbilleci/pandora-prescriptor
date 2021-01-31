QUANTIZED_FILE = "pandora/quantized_model.tflite"

# test with ~90 days
START_DATE = "2020-08-01"
END_DATE = "2020-11-01"

C1_MAX = 3
C2_MAX = 3
C3_MAX = 2
C4_MAX = 4
C5_MAX = 2
C6_MAX = 3
C7_MAX = 2
C8_MAX = 4
H1_MAX = 2
H2_MAX = 3
H3_MAX = 2
H6_MAX = 4

NPI_LIMITS = [C1_MAX,
              C2_MAX,
              C3_MAX,
              C4_MAX,
              C5_MAX,
              C6_MAX,
              C7_MAX,
              C8_MAX,
              H1_MAX,
              H2_MAX,
              H3_MAX,
              H6_MAX]
NPI_LIMITS_SIZE = len(NPI_LIMITS)

C1 = "C1_School closing"
C2 = "C2_Workplace closing"
C3 = "C3_Cancel public events"
C4 = "C4_Restrictions on gatherings"
C5 = "C5_Close public transport"
C6 = "C6_Stay at home requirements"
C7 = "C7_Restrictions on internal movement"
C8 = "C8_International travel controls"
H1 = "H1_Public information campaigns"
H2 = "H2_Testing policy"
H3 = "H3_Contact tracing"
H6 = "H6_Facial Coverings"
