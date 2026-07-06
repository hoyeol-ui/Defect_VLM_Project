import os
import xml.etree.ElementTree as ET


def parse_neu_xml(xml_path):
    """NEU-DET XML 파일을 읽어 결함 이름과 좌표를 반환"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        objects.append({
            "label": name,
            "bbox": [ymin, xmin, ymax, xmax]  # Florence-2는 [y1, x1, y2, x2] 형식을 선호함
        })
    return objects


if __name__ == "__main__":
    # 테스트 실행
    sample_xml = "data/NEU-DET/train/annotations/scratches_1.xml"
    if os.path.exists(sample_xml):
        data = parse_neu_xml(sample_xml)
        print(f"✅ 파싱 결과: {data}")
    else:
        print("XML 파일을 찾을 수 없습니다.")