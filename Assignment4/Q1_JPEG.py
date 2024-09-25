import numpy as np 
import matplotlib.pyplot as plt
import cv2
import sys
def DCT(f):
    Cu = np.zeros((8, 8))
    Cv = np.zeros((8, 8))
    output = np.zeros((8, 8))
    for u in range(8):
        for v in range(8):
            sum = 0
            for x in range(8):
                for y in range(8):
                    sum += f[x][y] * np.cos(((2.0*x+1)*u*np.pi)/16.0) * np.cos(((2.0*y+1)*v*np.pi)/16.0)

            Cu[u][v] = 1/np.sqrt(2) if u == 0 else 1
            Cv[u][v] = 1/np.sqrt(2) if v == 0 else 1
            output[u][v] = 1/4.0 * Cu[u][v] * Cv[u][v] * sum
            # print("%8.1f " % output[u][v], end="")
    return output

def IDCT(F):
    Cu = np.zeros((8, 8))
    Cv = np.zeros((8, 8))
    f = np.zeros((8, 8))
    for x in range(8):
        for y in range(8):
            sum = 0
            for u in range(8):
                for v in range(8):
                    Cu[u][v] = 1/np.sqrt(2) if u == 0 else 1
                    Cv[u][v] = 1/np.sqrt(2) if v == 0 else 1
                    sum += Cu[u][v] * Cv[u][v] * F[u][v] * np.cos(((2*x+1)*u*np.pi)/16) * np.cos(((2*y+1)*v*np.pi)/16)
            f[x][y] = 1/4 * sum
    return f

def RLE(arr):
    encoded_list = []
    current_value = arr[0]
    current_count = 1
    for i in range(1, len(arr)):
        if arr[i] == current_value:
            current_count += 1
        else:
            encoded_list.append((current_value, current_count))
            current_value = arr[i]
            current_count = 1
    encoded_list.append((current_value, current_count))
    return encoded_list

def RLD(tuples_list):
    decoded_values = []
    for value, run in tuples_list:
        decoded_values.extend([value] * run)
    return decoded_values

def Huffencode(num):
        num=int(num)
        if num<0:
            binary_representation = bin(-num)[2:]
            ones_complement = ''.join('0' if bit == '1' else '1' for bit in binary_representation)
            ones_string = '1' * len(binary_representation)
            result = ones_string + '0' + ones_complement
       
        elif num>0:
            binary_representation = bin(num)[2:]
            ones_string = '1' * len(binary_representation)
            result = ones_string + '0' + binary_representation
        
        else:
            result='0'
        # print(result)
        return result
def Huffdecode(string):
    # If the encoded string is '0', return 0
    if string == '0':
        return 0
    else:
        i=0
        while string[i] != '0':
               i+=1
            
        bin=string[i+1:]   
        if int(bin[0])==1:
            return int(bin, 2) 
        else:
            ones_complement = ''.join('0' if bit == '1' else '1' for bit in bin)    
            return -int(ones_complement, 2)

def zigzag_traversal(matrix):
    rows, cols = matrix.shape
    result = []

    # Start from top-left corner
    row, col = 0, 0
    going_up = True

    while row < rows and col < cols:
        result.append(matrix[row][col])
        if going_up:
            if col == cols - 1:
                row += 1
                going_up = False
            elif row == 0:
                col += 1
                going_up = False
            else:
                row -= 1
                col += 1
        else:
            if row == rows - 1:
                col += 1
                going_up = True
            elif col == 0:
                row += 1
                going_up = True
            else:
                row += 1
                col -= 1
    return result    

def zigzag_to_matrix(array):
    matrix = np.zeros((8, 8), dtype=int)
    zigzag_pattern = [
        (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
        (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
        (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
        (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
        (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
        (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
        (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
        (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
    ]
    for i, (row, col) in enumerate(zigzag_pattern):
        matrix[row, col] = array[i]
    return matrix
def jpeg_cv2(image):
    image= cv2.imread("C:/Users/nilad/Downloads/AIP asgmt4/cameraman.tif",cv2.IMREAD_GRAYSCALE)
    jpeg_quality = 50
    encoded_image = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])[1]
    jpeg_data = encoded_image.tobytes()
    bitstream = ''.join(format(byte, '08b') for byte in jpeg_data)
    # print(bitstream)
    file_path = "C:/Users/nilad/Downloads/AIP asgmt4/bits_with_CV.txt"
    with open(file_path, "w") as file:
        file.write(bitstream)
    binary_data = bytes(int(bitstream[i:i+8], 2) for i in range(0, len(bitstream), 8))

    with open('compressed_image.jpg','wb') as file:
        file.write(binary_data)
    compressed_image = cv2.imread('compressed_image.jpg',cv2.IMREAD_GRAYSCALE)
    return compressed_image,len(bitstream)
    # decompressed_image = cv2.imdecode(np.frombuffer(binary_data, np.uint8), cv2.IMREAD_COLOR)


if __name__=="__main__":
    input_image= cv2.imread("C:/Users/nilad/Downloads/AIP asgmt4/cameraman.tif",cv2.IMREAD_GRAYSCALE)
    input_image=input_image.astype(np.float64)-128.0
    # print(np.max(input_image))
    height, width = input_image.shape[:2]
    num_blocks_height = height // 8
    num_blocks_width = width // 8
    blocks = np.zeros((num_blocks_height, num_blocks_width, 8, 8), dtype=input_image.dtype)

    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            block = input_image[i*8:(i+1)*8, j*8:(j+1)*8]
            blocks[i, j] = block
    DCT_blocks=np.zeros((num_blocks_height, num_blocks_width, 8, 8), dtype=input_image.dtype)
    Q_blocks=np.zeros((num_blocks_height, num_blocks_width, 8, 8), dtype=input_image.dtype)
    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            DCT_blocks[i,j]=DCT(blocks[i,j])
    # print(DCT_blocks[0,0])
    # print(blocks[0,0])
    Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                   [12, 12, 14, 19, 26, 58, 60, 55],
                   [14, 13, 16, 24, 40, 57, 69, 56],
                   [14, 17, 22, 29, 51, 87, 80, 62],
                   [18, 22, 37, 56, 68,109, 103, 77],
                   [24, 35, 55, 64, 81, 104, 113, 92],
                   [49, 64, 78, 87, 103, 121, 120, 101],
                   [72, 92, 95, 98, 112, 100, 103, 99]])
    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            Q_blocks[i,j]=np.floor(DCT(blocks[i,j])/Q + 0.5)             
    # print(Q_blocks[0,0])
    zz=np.zeros((num_blocks_height,num_blocks_width,64))
    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            zz[i,j,:]=zigzag_traversal(Q_blocks[i,j])
    zzz=zz.reshape(-1,64)
          
    for i in range(num_blocks_height*num_blocks_width -1,0,-1):
            zzz[i,0]-= zzz[i-1,0]

    run_enc=[]
    for i in range(num_blocks_height*num_blocks_width):
            run_enc.append(RLE(zzz[i,:]))   
    
    huff_enc=''
    for x in run_enc:
        y=[(Huffencode(value), bin(runlength)[2:]) for value, runlength in x]
        encoded_string = ','.join(['{},{}'.format(bit_string, value) for bit_string, value in y])
        huff_enc+=(encoded_string)+','
    print("Compression Ratio:",height*width*8/len(huff_enc) )
    
    file_path = "C:/Users/nilad/Downloads/AIP asgmt4/bits_with_RLE.txt"
    with open(file_path, "w") as file:
        file.write(huff_enc)
    # print("String has been written to the file.")
    
    components = huff_enc.split(',')
    tuples_list = []
    for i in range(0, len(components)-2, 2):
        bit_string = components[i]
        occurrence_count = (components[i + 1])
        tuples_list.append((Huffdecode(bit_string), int(occurrence_count,2)))
    
    # y=[(Huffdecode(value), runlength) for value, runlength in tuples_list]
    run_dec=RLD(tuples_list)
    num_arrays = int(len(run_dec) / 64)
    # print(num_arrays)
    arrays = []
    for i in range(num_arrays):
        chunk = run_dec[i * 64 : (i + 1) * 64]
        arrays.append(np.array(chunk))
    # print(arrays[0])
    for i in range(1,num_arrays):
        arrays[i][0]+=arrays[i-1][0]
    matrices=[]
    for i in range(num_arrays):
       z= zigzag_to_matrix(arrays[i])
       z=z*Q
       matrices.append(IDCT(z))
    # print("hi")
    rec=np.zeros((256,256))  
    k=0 
    # print(len(matrices))
    for i in range(32):
        for j in range(32):
            # print(k)
            rec[i*8:(i+1)*8, j*8:(j+1)*8]=matrices[k]
            k+=1
    # print(np.max(rec),np.min(rec))        
    rec+=128 
    rec=np.clip(rec,0,255)
    # print(np.max(rec),np.min(rec))
    input_image+=128
    # print(np.max(input_image),np.min(input_image))
    
    rec2,l=jpeg_cv2(input_image)
    print("Mean Square Error with our implementation:", np.sum((rec-input_image)**2)/(256*256))
    print("Mean Square Error with CV2's implementation:", np.sum((rec2-input_image)**2)/(256*256))
    print("File size(in bits): Ours:",len(huff_enc),"CV2's:",l)
    fig, axes = plt.subplots(1, 2)  
    axes[0].imshow(rec, cmap='gray')
    axes[0].set_title('Image 1')
    axes[0].axis('off')
    axes[1].imshow(rec2, cmap='gray')
    axes[1].set_title('Image 2')
    axes[1].axis('off')
    plt.show()


    
    
    


    
    
