from pandora.quantized_constants import C1_MAX, C2_MAX, C3_MAX, C4_MAX, C5_MAX, C6_MAX, \
    C7_MAX, C8_MAX, H1_MAX, H2_MAX, H3_MAX, H6_MAX


def generate_initial_points(number_of_days):
    if number_of_days == 1:
        return generate_initial_points_for_segment(number_of_days)
    else:
        segment_1 = generate_initial_points_for_segment(int(number_of_days / 2))
        segment_2 = generate_initial_points_for_segment(number_of_days - int(number_of_days / 2))
        joined = []
        for x1 in segment_1:
            for x2 in segment_2:
                joined.append(x1 + x2)
        return joined


def generate_initial_points_for_segment(number_of_days):
    initial_min = number_of_days * 12 * [0]
    initial_one = number_of_days * 12 * [1]
    initial_two = number_of_days * 12 * [2]
    initial_max = number_of_days * [C1_MAX, C2_MAX, C3_MAX, C4_MAX,
                                    C5_MAX, C6_MAX, C7_MAX, C8_MAX,
                                    H1_MAX, H2_MAX, H3_MAX, H6_MAX]
    initial_points = [initial_min,
                      initial_one,
                      initial_two,
                      initial_max]
    initial_points += generate_initial_points_for_npi(C1_MAX, 0, number_of_days)
    initial_points += generate_initial_points_for_npi(C2_MAX, 1, number_of_days)
    initial_points += generate_initial_points_for_npi(C3_MAX, 2, number_of_days)
    initial_points += generate_initial_points_for_npi(C4_MAX, 3, number_of_days)
    initial_points += generate_initial_points_for_npi(C5_MAX, 4, number_of_days)
    initial_points += generate_initial_points_for_npi(C6_MAX, 5, number_of_days)
    initial_points += generate_initial_points_for_npi(C7_MAX, 6, number_of_days)
    initial_points += generate_initial_points_for_npi(C8_MAX, 7, number_of_days)
    initial_points += generate_initial_points_for_npi(H1_MAX, 8, number_of_days)
    initial_points += generate_initial_points_for_npi(H2_MAX, 9, number_of_days)
    initial_points += generate_initial_points_for_npi(H3_MAX, 10, number_of_days)
    initial_points += generate_initial_points_for_npi(H6_MAX, 11, number_of_days)
    return initial_points


def generate_initial_points_for_npi(npi_max: int,
                                    npi_position: int,
                                    number_of_days: int):
    initial_points = []
    for i in range(0, npi_max + 1):
        if i > 0:
            x = number_of_days * [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            x[npi_position] = i
            initial_points.append(x)
        if i < C1_MAX:
            x = number_of_days * [C1_MAX, C2_MAX, C3_MAX, C4_MAX,
                                  C5_MAX, C6_MAX, C7_MAX, C8_MAX,
                                  H1_MAX, H2_MAX, H3_MAX, H6_MAX]
            x[npi_position] = i
            initial_points.append(x)
    return initial_points
