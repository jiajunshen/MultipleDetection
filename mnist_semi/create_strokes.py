import numpy as np

def generate(images, mega_patch_w, result_sample_w, num_strokes, num_samples):
    image_size, image_w, image_h = images.shape
    image_channel = 1
    generated_image = np.zeros((num_samples, result_sample_w, result_sample_w))

    for j in range(num_samples):
        for i in range(num_strokes):
            intensity = 0
            t_y = np.random.randint(result_sample_w - mega_patch_w)
            t_x = np.random.randint(result_sample_w - mega_patch_w)
            patch = np.zeros((mega_patch_w, mega_patch_w))
            while(intensity < 10):
                index = np.random.randint(image_size)
                s_y = np.random.randint(image_w - mega_patch_w)
                s_x = np.random.randint(image_h - mega_patch_w)
                patch = np.maximum(images[index, s_x:s_x + mega_patch_w, s_y:s_y + mega_patch_w], patch)
                intensity = np.sum(patch)
        
            #generated_image[j, t_x:t_x + mega_patch_w, t_y:t_y + mega_patch_w] = np.maximum(patch, generated_image[j, t_x:t_x + mega_patch_w, t_y:t_y + mega_patch_w])
            generated_image[j, t_x:t_x + mega_patch_w, t_y:t_y + mega_patch_w] = patch

    return generated_image


def main():
    X = np.load("/Users/jiajunshen/.mnist/X_train.npy")
    result = generate(X, 8, 60, 5, 50)
    print(result.shape)
    np.save("../data/stroke.npy", result)

if __name__ == "__main__":
    main()
    
    
