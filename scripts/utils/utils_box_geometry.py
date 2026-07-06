from typing import Dict, List


def clip_xyxy(
    xyxy: List[float],
    image_width: int,
    image_height: int
) -> List[float]:
    """
    Clip bbox coordinates to image boundary.

    OVD models sometimes return coordinates outside image boundaries.
    Example:
        y_min < 0 or x_max > image_width

    This function makes pseudo boxes valid before geometry calculation.
    """
    xmin, ymin, xmax, ymax = map(float, xyxy)

    xmin = max(0.0, min(xmin, image_width - 1))
    ymin = max(0.0, min(ymin, image_height - 1))
    xmax = max(0.0, min(xmax, image_width - 1))
    ymax = max(0.0, min(ymax, image_height - 1))

    # 혹시 좌표 순서가 뒤집힌 경우 방어
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin

    return [xmin, ymin, xmax, ymax]


def xyxy_to_geometry(
    xyxy: List[float],
    image_width: int,
    image_height: int
) -> Dict:
    """
    Convert absolute xyxy bbox coordinates into normalized geometry attributes.

    Used for both:
    1) GT bounding boxes from VOC XML
    2) pseudo bounding boxes from open-vocabulary detectors

    Returns:
        raw_bbox
        norm_bbox
        center
        width_ratio
        height_ratio
        area_ratio
        aspect_ratio
        spatial_zone
        scale_category
    """
    xmin, ymin, xmax, ymax = clip_xyxy(xyxy, image_width, image_height)

    box_w = max(0.0, xmax - xmin)
    box_h = max(0.0, ymax - ymin)

    width_ratio = round(box_w / image_width, 4)
    height_ratio = round(box_h / image_height, 4)

    norm_xmin = round(xmin / image_width, 4)
    norm_ymin = round(ymin / image_height, 4)
    norm_xmax = round(xmax / image_width, 4)
    norm_ymax = round(ymax / image_height, 4)

    center_x = round((norm_xmin + norm_xmax) / 2.0, 4)
    center_y = round((norm_ymin + norm_ymax) / 2.0, 4)

    area_ratio = round(width_ratio * height_ratio, 4)

    # width / height. 세로 scratch는 aspect_ratio가 작을 수 있음.
    aspect_ratio = round(box_w / box_h, 4) if box_h > 0 else 999.0

    v_pos = "center" if 0.33 <= center_y <= 0.66 else ("top" if center_y < 0.33 else "bottom")
    h_pos = "center" if 0.33 <= center_x <= 0.66 else ("left" if center_x < 0.33 else "right")

    spatial_zone = "center" if v_pos == "center" and h_pos == "center" else f"{v_pos}-{h_pos}"

    # 기존 3단계 유지: downstream groundedness 계산 단순화를 위해 그대로 둠
    scale_category = (
        "micro" if area_ratio < 0.01
        else "small" if area_ratio < 0.05
        else "large"
    )

    return {
        "raw_bbox": [round(xmin, 2), round(ymin, 2), round(xmax, 2), round(ymax, 2)],
        "norm_bbox": [norm_xmin, norm_ymin, norm_xmax, norm_ymax],
        "center": [center_x, center_y],
        "width_ratio": width_ratio,
        "height_ratio": height_ratio,
        "area_ratio": area_ratio,
        "aspect_ratio": aspect_ratio,
        "spatial_zone": spatial_zone,
        "scale_category": scale_category
    }


def compute_box_quality(
    score: float,
    geometry: Dict,
    area_penalty_weight: float = 1.0
) -> float:
    """
    Compute pseudo box quality score.

    Motivation:
        OVD confidence alone is not enough.
        Very large boxes often cover background/surface regions rather than defects.

    Formula:
        quality = score * (1 - area_ratio) ^ area_penalty_weight

    Higher:
        high confidence + local box

    Lower:
        low confidence or overly large box
    """
    area_ratio = float(geometry.get("area_ratio", 0.0))

    area_factor = max(0.0, 1.0 - area_ratio)
    quality = float(score) * (area_factor ** area_penalty_weight)

    return round(quality, 6)


def is_valid_pseudo_box(
    geometry: Dict,
    min_area_ratio: float = 0.0001,
    max_area_ratio: float = 0.20,
    max_width_ratio: float = 0.90,
    max_height_ratio: float = 0.95
) -> bool:
    """
    Filter invalid or overly broad pseudo boxes.

    Industrial defects are usually local or elongated local regions.
    Very large boxes often indicate that OVD captured the whole surface/image
    rather than the defect itself.

    Args:
        geometry: output of xyxy_to_geometry()
        min_area_ratio: remove tiny noise boxes
        max_area_ratio: remove overly large surface-level boxes
        max_width_ratio: remove boxes covering almost full width
        max_height_ratio: remove boxes covering almost full height

    Note:
        For scratch defects, elongated boxes may be valid.
        Therefore, this function does not filter by aspect ratio directly.
    """
    area = float(geometry["area_ratio"])
    width_ratio = float(geometry["width_ratio"])
    height_ratio = float(geometry["height_ratio"])

    if area < min_area_ratio:
        return False

    if area > max_area_ratio:
        return False

    if width_ratio > max_width_ratio:
        return False

    if height_ratio > max_height_ratio:
        return False

    return True


if __name__ == "__main__":
    test_boxes = [
        [1.0, -1.54, 2044.98, 987.58],
        [1573.04, 192.44, 2004.57, 251.35],
        [27.56, 52.01, 60.34, 199],
    ]

    for box in test_boxes:
        geom = xyxy_to_geometry(
            xyxy=box,
            image_width=2048,
            image_height=1000
        )
        print("\n[TEST BOX]", box)
        print(geom)
        print("valid:", is_valid_pseudo_box(geom))
        print("quality:", compute_box_quality(0.02, geom))