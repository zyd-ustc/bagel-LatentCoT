"""Build explicit semantic tests plus safely split held-out v1 captions."""
import argparse
import json
from pathlib import Path

# The middle sentence is withheld from GEN. UND gets all three sentences.
SCENES = [
('books','counting', 'A wooden reading table stands in a quiet library, viewed straight on in clear daylight, with a brass desk lamp on its left and a plain wall behind it.', 'Exactly three closed books are stacked on the right side of the table: a red book at the bottom, a green book in the middle, and a blue book on top.', 'Leave ample empty space between the lamp and the book stack, with every book edge clearly visible and no additional books anywhere in the scene.', ['书是否恰好三本','从下到上是否红、绿、蓝']),
('sunflowers','counting', 'A simple white ceramic vase stands in the center of a pale wooden table in front of a flat light gray wall, photographed at eye level in soft daylight.', 'The vase contains exactly three sunflowers, each with a distinct dark brown center and yellow petals, and all three flower heads are fully visible without overlapping.', 'Use a clean uncluttered arrangement with no other flowers, plants, pictures, or patterned decorations in the background.', ['是否恰好三个向日葵花盘','是否全部在同一个白色花瓶内']),
('birds','counting', 'A wide wooden shelf is mounted against a plain cream wall, with a small empty white bowl near its left edge, photographed from the front in bright soft light.', 'Exactly four small blue birds stand in a row on the right half of the shelf, and each bird has a bright yellow beak.', 'Separate the birds so every body and beak can be counted clearly, and keep the rest of the room empty and neutral.', ['鸟是否恰好四只','是否蓝色身体、黄色喙']),
('cups','binding', 'A square dining table covered with a plain white cloth is photographed from a slightly elevated angle, with a small green plant centered along the back edge.', 'Place a red teacup on a blue saucer to the left of the plant and a blue teacup on a red saucer to the right of the plant.', 'The two cup and saucer pairs must be separated by empty table space, under even lighting with no patterned tableware or other dishes.', ['左侧是否红杯蓝碟','右侧是否蓝杯红碟']),
('umbrellas','binding', 'Two adults stand far apart on a broad wet pavement in front of a plain gray wall, shown as full body figures facing the camera in an uncluttered scene.', 'The person on the left wears a yellow coat and holds a purple umbrella, while the person on the right wears a purple coat and holds a yellow umbrella.', 'Keep both umbrellas fully open and separated, and show the coat colors clearly without rain obscuring the figures.', ['左侧是否黄衣紫伞','右侧是否紫衣黄伞']),
('fruit','binding', 'A white rectangular tray rests on a pale kitchen counter, viewed from above in clear neutral daylight, with an empty glass placed outside the tray near its top edge.', 'Inside the tray, arrange exactly two red apples on the left and exactly three yellow pears on the right, keeping the two fruit groups apart.', 'All five pieces of fruit should be fully visible and naturally shaped, with no labels, extra fruit, or decorative clutter.', ['是否左边两只红苹果','是否右边三只黄梨']),
('chair','spatial', 'A single green upholstered chair stands in a bright studio with a pale floor and a plain white wall, viewed directly from the front with plenty of space around it.', 'A red ball lies on the floor to the left of the chair, and a blue square cushion lies on the floor to the right of the chair.', 'Keep the ball and cushion clearly separate from the chair, using realistic soft shadows and no additional furniture or objects.', ['红球是否在椅子左边','蓝色方形靠垫是否在椅子右边']),
('shelf','spatial', 'A freestanding white shelving unit with two wide horizontal shelves stands against a pale wall, photographed directly from the front in clear soft daylight.', 'Place a small red toy car on the upper shelf and a large blue toy truck on the lower shelf, both facing toward the right edge of the image.', 'Show both shelves completely and leave empty space around each vehicle so the vertical placement and direction are unambiguous.', ['上层是否红色小汽车、下层是否蓝色卡车','两辆车是否均朝右']),
('cat','spatial', 'A long wooden park bench is shown from the front in a quiet sunny garden, with a smooth path in the foreground and a simple green hedge behind it.', 'A white cat sits under the middle of the bench, while a black dog sits on top of the bench near its right end.', 'Keep both animals fully visible with clear separation from the bench legs, and include no other animals or people.', ['白猫是否在长椅下方','黑狗是否在长椅上方右侧']),
('hats','binding', 'Three people stand in a straight line in front of a plain light gray studio backdrop, shown from the waist up with enough space to distinguish each person clearly.', 'From left to right, their hats are red, green, and blue; the person in the middle also holds a small yellow flower at chest level.', 'Use neutral plain shirts and soft even lighting, and keep all three hats and the flower completely visible.', ['帽子从左到右是否红绿蓝','是否仅中间的人拿黄色花']),
('bottles','counting', 'A narrow white display pedestal stands in a minimalist gallery with a pale gray wall behind it, photographed straight on with clean soft lighting and clear edges.', 'Exactly five transparent glass bottles sit on the pedestal in one row, with the central bottle filled with green liquid and the other four completely empty.', 'Leave small gaps between the bottles, show all five bottle necks distinctly, and avoid reflections that look like additional bottles.', ['是否恰好五个瓶子','是否仅中间一瓶有绿色液体']),
('train','spatial', 'A small wooden toy train rests on a plain tabletop, shown from the side against a neutral cream background, with all wheels and carriages visible in soft daylight.', 'The locomotive is bright red and pulls exactly two carriages: a yellow carriage directly behind it and a blue carriage at the very back.', 'Keep the train in a straight horizontal line, with no extra carriages, toy buildings, people, or decorative objects.', ['是否红色车头、恰好两节车厢','是否依次黄色车厢、蓝色车厢']),
('sign','text', 'A small bakery storefront faces the camera on a quiet pedestrian street, with a wooden door, two simple display windows, and a clean pale stone facade in morning light.', 'A single dark blue sign above the door displays the exact words BLUE MOON in large white uppercase letters, with no other words on the sign.', 'Make the sign flat, legible, and fully visible, and avoid posters, window lettering, or other signs that distract from the main text.', ['招牌是否准确写 BLUE MOON','是否蓝底白色大写字母']),
('kite','binding', 'Two children stand apart on a broad grassy hill under a clear pale blue sky, shown as complete figures with their arms and kite strings visible in bright daylight.', 'The child on the left wears a red shirt and flies a blue diamond shaped kite, while the child on the right wears a blue shirt and flies a red diamond shaped kite.', 'Keep the two kites high in the sky and well separated, with no other people, kites, or flying objects.', ['左边是否红衣蓝风筝','右边是否蓝衣红风筝']),
('candle','counting', 'A round white birthday cake sits on a plain blue cake stand in front of a neutral gray wall, photographed from a slightly elevated angle in bright even lighting.', 'Exactly four slim yellow candles stand upright on the cake, arranged in a square, with each candle burning a small visible flame.', 'Keep the cake undecorated apart from the candles, make every candle easy to count, and add no flowers, fruit, numbers, or written text.', ['是否恰好四根黄色蜡烛','是否四根均有火焰并排列成方形']),
('rabbit','binding', 'Two rabbits sit far apart on a simple wooden floor against a pale beige wall, shown from the front in soft daylight with both faces and all ears fully visible.', 'The white rabbit on the left wears a blue ribbon around its neck, and the brown rabbit on the right wears a red ribbon around its neck.', 'Keep the ribbons large and clearly colored, with no additional animals, toys, furniture, or patterned background.', ['左边是否白兔蓝丝带','右边是否棕兔红丝带']),
]


def build_authored():
    return [dict(id='semantic_'+name,group='controlled_long',category=category,
                 full_prompt=' '.join((first,missing,last)),partial_prompt=first+' '+last,
                 removed_clauses=missing,checks=checks,source='authored_compositional_v1')
            for name,category,first,missing,last,checks in SCENES]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--heldout-count',type=int,default=8)
    args=parser.parse_args()
    from qwen_latent_cot.interaction.data import GenerationCorpus
    from qwen_latent_cot.interaction.stage_a import split_prompt
    rows=build_authored()
    corpus=GenerationCorpus(args.corpus)
    families=set(); heldout=[]
    views=corpus.groups('val')
    for pos in range(2048):
        for view in views:
            if pos>=len(view):continue
            row=view[pos];full=row['caption']
            if row['family_id'] in families or not 45<=len(full.split())<=160 or any(x in full for x in ('###','**','<','>','\\n')):continue
            split=split_prompt(full,17+len(heldout),full_probability=0,min_words=30)
            if not split.removed:continue
            heldout.append(dict(id='heldout_'+str(len(heldout)),group='heldout_long',category='v1_caption',
                                full_prompt=full,partial_prompt=split.partial,removed_clauses=split.removed,
                                checks=['检查删掉的句子中，对象、属性、数量和关系是否得到恢复'],
                                source=row['source'],family_id=row['family_id'],source_sample_id=row['sample_id']))
            families.add(row['family_id'])
            if len(heldout)==args.heldout_count:break
        if len(heldout)==args.heldout_count:break
    if len(heldout)!=args.heldout_count:raise ValueError('Not enough safely split held-out v1 captions')
    rows+=heldout
    short=[json.loads(line) for line in Path('configs/training_free_prompts_64.jsonl').read_text().splitlines() if line.strip()][:8]
    for i,row in enumerate(short):
        rows.append(dict(id=f'short_{i}',group='short_control',category='short',full_prompt=row['prompt'],
                         partial_prompt=row['prompt'],removed_clauses='',checks=['短 Prompt 是否被破坏；比较清晰度和要求符合情况'],source='training_free_prompts_64'))
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    corpus.close()
    print(json.dumps(dict(output=str(output),cases=len(rows),controlled=16,heldout=len(heldout),short=len(short))))


if __name__=='__main__':main()
