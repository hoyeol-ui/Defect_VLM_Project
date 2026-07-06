import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional

# GC10-DET 클래스 정보 매핑 사전 (중국어 성조 태그 -> 표준 영문 클래스)
GC10_LABEL_MAP = {
    "1_chongkong": "punching_hole",
    "2_hanfeng": "welding_line",
    "3_yueyawan": "crescent_gap",
    "4_shuiban": "water_spot",
    "5_youban": "oil_spot",
    "6_siban": "silk_spot",
    "7_yiwu": "inclusion",
    "8_yahen": "rolled_pit",
    "9_zhehen": "crease",
    "10_yaozhe": "waist_folding"
}


def parse_voc_xml(xml_path: Any, dataset_type: str = "NEU-DET") -> Dict[str, Any]:
    """
    PASCAL VOC 형식의 XML 파일을 파싱하여 이미지 규격 정보와
    바운딩 박스(BBox) 리스트를 자연어 지표를 포함한 딕셔너리로 반환합니다.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"XML 파일을 찾을 수 없습니다: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_node = root.find("size")
    width = int(size_node.find("width").text)
    height = int(size_node.find("height").text)

    objects = []
    for obj in root.findall("object"):
        raw_label = obj.find("name").text

        if "GC10" in dataset_type.upper():
            label_name = GC10_LABEL_MAP.get(raw_label, raw_label)
        else:
            label_name = raw_label

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        norm_xmin = round(xmin / width, 4)
        norm_ymin = round(ymin / height, 4)
        norm_xmax = round(xmax / width, 4)
        norm_ymax = round(ymax / height, 4)

        center_x = round((norm_xmin + norm_xmax) / 2, 4)
        center_y = round((norm_ymin + norm_ymax) / 2, 4)
        bbox_area_ratio = round((norm_xmax - norm_xmin) * (norm_ymax - norm_ymin), 4)

        v_pos = "center" if 0.33 <= center_y <= 0.66 else ("top" if center_y < 0.33 else "bottom")
        h_pos = "center" if 0.33 <= center_x <= 0.66 else ("left" if center_x < 0.33 else "right")
        spatial_zone = f"{v_pos}-{h_pos}" if v_pos != "center" or h_pos != "center" else "center"

        scale_category = "micro" if bbox_area_ratio < 0.01 else ("small" if bbox_area_ratio < 0.05 else "large")

        objects.append({
            "label": label_name,
            "raw_label": raw_label,
            "raw_bbox": [xmin, ymin, xmax, ymax],
            "norm_bbox": [norm_xmin, norm_ymin, norm_xmax, norm_ymax],
            "center": [center_x, center_y],
            "area_ratio": bbox_area_ratio,
            "spatial_zone": spatial_zone,
            "scale_category": scale_category
        })

    return {
        "filename": root.find("filename").text,
        "width": width,
        "height": height,
        "objects": objects
    }


def get_xml_path_from_image(image_path: Any, dataset_type: str = "NEU-DET") -> Optional[Path]:
    """
    실제 절대 경로 구조 및 'lable' 오탈자 폴더를 고려하여
    이미지에 매칭되는 XML 라벨 파일의 경로를 추적합니다.
    """
    image_path = Path(image_path)
    img_stem = image_path.stem
    project_root = Path("/Users/hy/PycharmProjects/PythonProject/Defect_VLM_Project")

    if "NEU" in dataset_type.upper():
        xml_path = project_root / "data" / "NEU-DET" / "ANNOTATIONS" / f"{img_stem}.xml"
        if xml_path.exists():
            return xml_path
    elif "GC10" in dataset_type.upper():
        xml_path = project_root / "data" / "GC10-DET" / "lable" / f"{img_stem}.xml"
        if xml_path.exists():
            return xml_path

    return None


if __name__ == "__main__":
    print("[경로 검증 동적 탐색 테스트]")
    project_root = Path("/Users/hy/PycharmProjects/PythonProject/Defect_VLM_Project")

    # 1. NEU-DET 테스트
    neu_img_dir = project_root / "data" / "NEU-DET" / "IMAGES"
    neu_images = list(neu_img_dir.glob("*.jpg"))
    if neu_images:
        print(f"  -> 발견된 NEU-DET 이미지: {neu_images[0].name}")
        xml_p = get_xml_path_from_image(neu_images[0], dataset_type="NEU-DET")
        if xml_p and xml_p.exists():
            res = parse_voc_xml(xml_p, dataset_type="NEU-DET")
            print(f"  ✅ 파싱 성공: {res['objects'][0]['label']} | 구역: {res['objects'][0]['spatial_zone']}")

    # 2. GC10-DET 테스트
    gc_root_dir = project_root / "data" / "GC10-DET"
    gc10_images = list(gc_root_dir.glob("**/img_*.jpg"))
    if gc10_images:
        print(f"  -> 발견된 GC10-DET 이미지: {gc10_images[0].name}")
        xml_p = get_xml_path_from_image(gc10_images[0], dataset_type="GC10-DET")
        if xml_p and xml_p.exists():
            res = parse_voc_xml(xml_p, dataset_type="GC10-DET")
            print(f"  ✅ 파싱 성공: {res['objects'][0]['label']} | 구역: {res['objects'][0]['spatial_zone']}")