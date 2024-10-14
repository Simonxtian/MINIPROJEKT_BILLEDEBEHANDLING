import cv2
import numpy as np
from collections import deque

image = cv2.imread('4.jpg') #Read the image
r_image=cv2.resize(image, (500, 500)) #Resize så vi har styr på størrelsen

crown_template = cv2.imread('crown2.png', 0) #læs billedet
template_h, template_w = crown_template.shape[:2] #Få højde og bredde af billedet
crown_detection_results = [] #Gemmer resultaterne af kronedetektionen
threshold = 0.48 #til template matching
cropped_images = [] #tomt array til at store de croppede billeder

for i in range(5): #Itererer over rows
    for j in range(5): #Itererer over columns
        cropped_image = r_image[i*100:(i+1)*100, j*100:(j+1)*100]
        cropped_images.append(cropped_image) #Gemmer de croppede billeder i en liste

        hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV) #Convert det cropped billede til HSV
        avg_hsv_per_row = np.average(hsv_image, axis=0) #udregner gennemstitlige HSV værdier per row


#Predefined HSV katagorier [H, S, V] - Fra at loade billederne en af gangen og finde gennemsnittet af HSV værdierne
categories = {
    "Drylands": [
        [24.5162, 153.6771, 102.7741],
        [21.846, 153.98, 108.1528],
        [25.7012, 114.6189, 109.0208],
        [25.1282, 134.1846,  83.2193],
        [25.2381,  84.4957, 108.8012],
        [23.9685, 125.1573,  98.4476],
        [23.6735, 130.9151,  96.3389],
        [22.1875, 164.6013, 106.7374],
        [25.3924, 117.1441, 106.7516],
        [23.7861, 128.2473, 131.351 ]
    ],
    "Planes": [
        [45.774, 216.3725, 144.5484],
        [44.7579, 208.2183, 150.2215],
        [42.3962, 224.6446, 144.4186],
        [33.49, 200.7627, 129.3985],
        [34.7143, 214.9313, 103.8343]
    ],
    "Water": [
        [82.9691, 217.4779, 136.7086],
        [99.4852, 232.2103, 152.1692],
        [99.9303, 247.3077, 164.3532],
        [83.6345, 215.5869, 129.2461]
    ],
    "Woods": [
        [46.6315, 160.7326, 56.8426],
        [37.1104, 148.1173, 58.3045],
        [45.8039, 162.8936, 50.7721],
        [44.9355, 153.4888, 45.1187],
        [45.1864, 161.1531,  66.1181],
	    [43.6752, 152.9081,  61.6782],
	    [44.4531, 159.8208,  63.2256],
	    [44.8309, 160.4163,  61.8739],
	    [43.0294, 167.6024,  62.8192],
	    [36.5725, 158.8622,  70.603 ],
        [48.1779, 100.3718,  68.259 ],
        [42.1008, 130.9627,  67.3185],
        [39.1435, 112.6216, 74.8062],
        [43.0581, 119.4372,  70.0183]
    ],
    "Desert": [
        [25.7501, 233.121, 195.9712],
        [25.6625, 220.0196, 193.7912],
        [26.1769, 224.6549, 194.0987],
        [24.7006, 214.8766, 172.6576],
        [24.7609, 230.0303, 143.9958],
        [26.2128, 252.1913, 171.3473],
        [26.7016, 238.2054, 169.5491],
        [26.2521, 246.9018, 174.2723]
    ],
    "Mine": [
        [42.4726, 112.3309, 64.6378],
        [52.3009, 94.3992, 51.4024],
        [50.5659, 81.3502, 50.2716],
        [51.2224, 96.8871, 52.7262],
        [29.8934, 135.6631,  59.6385],
        [33.5659, 104.5056,  57.335 ],
        [32.1827, 106.5468,  63.9784],
        [28.6152, 120.8797,  74.8266],
        [39.0038, 113.6101,  66.4054],
        [46.2547, 99.5427, 66.3457]
    ],
    "Table": [
        [20.5622, 216.1424, 114.5804],
        [20.432,  210.4944, 121.4268],
        [23.524, 207.2546, 137.4861],
        [20.9953, 201.904,  142.3827],
        [ 20.6717, 199.2588, 145.2444],
        [ 20.7995, 198.0386, 143.2196]
    ],
    "Castle": [
        [27.9522, 120.889,  148.3814],
        [55.7214,  66.2775, 138.9117],
        [57.796,  67.8949, 120.1129],
        [36.6191, 74.5822, 93.5671],
        [28.2856, 116.738,  138.2781],
        [28.5602, 110.4772, 139.9255],
        [33.0977,  70.4231, 133.0123],
        [36.0072, 63.8489, 98.5188],
        [73.6141, 40.1446, 94.8333]

    ]
}

#Udregner så den euclidiske afstand mellem to HSV værdier
def calculate_distance(hsv1, hsv2):
    return np.sqrt(np.sum((np.array(hsv1) - np.array(hsv2)) ** 2))

#Funktion for at kategorisere et billede baseret på dets gennemsnitlige HSV-værdier
def categorize_image(avg_hsv):
    min_distance = 255 #bare noget højt (midlertidig værdi - må bare ikke blive false senere)

    # Iterate over each category and calculate the distance to predefined HSV values
    for category, hsv_values in categories.items(): #itererer over hver kategori og udregner afstanden til de foruddefinerede HSV værdier
        for hsv in hsv_values: #itererer over hver HSV værdi i kategorien
            distance = calculate_distance(avg_hsv, hsv) #Udregner afstanden mellem den gennemsnitlige HSV værdi i billedet og den foruddefinerede HSV værdi
            if distance < min_distance: #Hvis afstanden er mindre end den mindste afstand
                min_distance = distance #Sæt den mindste afstand til at være afstanden
                closest_category = category #gem kategorien som den tætteste kategori
    
    return closest_category #return closest category. Vi går ud fra at det er den katagori billedet er tættest på. PGA den euclidiske afstand


# Create a 5x5 array to store the categories
category_grid = np.empty((5, 5), dtype=object) #Putter det hele i et 5x5 array

#Fylder arrayet med kategorier
for i in range(5): #Itererer over rows
    for j in range(5): #Itererer over columns
        index = i * 5 + j
        category_grid[i, j] = categorize_image(np.average(np.average(cv2.cvtColor(cropped_images[index], cv2.COLOR_BGR2HSV), axis=0), axis=0)) #Fylder arrayet med kategorier

#Funktion til at finde naboer og isolerede strenge ved hjælp af grassfire algoritmen
def find_neighboring_similar_strings_grassfire(grid): #Funktionen tager et grid som input
    def grassfire(x, y, visited, category):
        queue = deque([(x, y)])  #Initier en kø med startpunktet (x, y)
        count = 0 #Tæller antallet af forbundne komponenter
        
        while queue: #Så længe der er elementer i køen
            cx, cy = queue.popleft()  #Her tager vi det første element i køen
            if (cx, cy) in visited: #Hvis det er besøgt
                continue #Så fortsætter vi
            
            visited.add((cx, cy))  #Markér cellen som besøgt
            count += 1  #Tæl cellen plus 1
            
            # Check alle fire neighbors (left, right, top, bottom)
            for nx, ny in [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]: #Itererer over alle naboer
                if (0 <= nx < len(grid) and #len(grid) er antallet af rækker i grid. Hvis nx er størrer end antallet af rækker i grid, er du ude for gridet
                    0 <= ny < len(grid[0]) and  #og y er inde for gridet og ikke er mindre end 0
                    grid[nx][ny] == category and #hvis naboen har samme kategori
                    (nx, ny) not in visited): #og ikke er besøgt endnu
                    queue.append((nx, ny))  #Tilføj nabo til køen
        
        return count #Returner antallet af forbundne komponenter

    visited = set()  #besøgte celler
    neighboring_similar_strings = []  #For clusters størrer end 1
    isolated_strings = []  #Isolerede strenge

    for i in range(len(grid)): #Itererer over alle celler i grid
        for j in range(len(grid[0])):
            if (i, j) not in visited:  #Hvis en celle ikke er besøgt
                size = grassfire(i, j, visited, grid[i][j])  #Start grassfire algoritmen
                if size > 1: #Hvis størrelsen er større end 1
                    neighboring_similar_strings.append((grid[i][j], size))  #Record neighboring similar strings
                else: #Ellers
                    isolated_strings.append(grid[i][j])  #Record isolated strings
    
    return neighboring_similar_strings, isolated_strings #Returnér de to lister af strenge


# Template matching for at finde crowns
for i in range(5): #Itererer over rows
    for j in range(5): #Itererer over columns
        cropped_image = r_image[i*100:(i+1)*100, j*100:(j+1)*100] #laver 25 billeder
        cropped_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY) #grayscale
        
        match_count = 0  # Initialize match count for each tile
        for k in range(4):  # Check for each 90 degree rotation
            rotated_template = np.rot90(crown_template, k)
            
            res = cv2.matchTemplate(cropped_gray, rotated_template, cv2.TM_CCOEFF_NORMED) #Template matching
            loc = np.where(res >= threshold) #Finder steder hvor match er større end threshold
            match_count += np.sum(res >= threshold) #Tæller antallet af matches fundet for denne rotation
        
        crown_detection_results.append((i, j, match_count)) #Gemmer resultaterne med match count



#Funktion til at tælle antallet af tiles og kroner i hver cluster
def count_tiles_and_crowns(grid, crown_results):
    def gf_crowns_and_tiles(x, y, visited, category):
        stack = [(x, y)]
        tile_count = 0
        crown_count = 0
        while stack: #ikke er tom
            cx, cy = stack.pop() #pop element
            if (cx, cy) in visited: #hvis elementet er besøgt
                continue #så fortsæt
            visited.add((cx, cy)) #markér elementet som besøgt
            tile_count += 1 #tæl tiles plus 1
            
            #Se så om der er en krone i den tile
            if crown_results[cx * 5 + cy][2]:  # crown_results contains (i, j, found)
                crown_count += 1 #plus en krone

            for nx, ny in [(cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)]: #Itererer over alle naboer (venstre, højre, top, bund)
                if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == category and (nx, ny) not in visited:
                    stack.append((nx, ny)) #Tilføj nabo til stacken
        return tile_count, crown_count #Returnér antallet af tiles og kroner

    visited = set() #besøgte celler
    results = [] #tom liste til at gemme resultaterne
    for i in range(len(grid)): #Itererer over alle celler i grid
        for j in range(len(grid[0])):
            if (i, j) not in visited: #Hvis cellen ikke er besøgt
                tile_count, crown_count = gf_crowns_and_tiles(i, j, visited, grid[i][j]) #Start grassfire algoritmen
                if crown_count > 0:  #Kun clusters med mere end 0 kroner
                    results.append((tile_count, crown_count, tile_count * crown_count)) #gange gange gange

    return results

crown_clusters_summary = count_tiles_and_crowns(category_grid, crown_detection_results) #kald funk


for index, (tiles, crowns, product) in enumerate(crown_clusters_summary): #Itererer over alle clusters
    total_product = sum(product for _, _, product in crown_clusters_summary) #Udregn det totale produkt af alle clusters

print(f"Total product of all clusters: {total_product}") #Print det totale produkt af alle clusters
