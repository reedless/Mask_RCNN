# Run bash container
docker run -it --rm -p 8888:8888 -p 6006:6006 -v ~/Mask_RCNN/:/host waleedka/modern-deep-learning

# Run jupyter notebook
# docker run -it --rm -p 8888:8888 -p 6006:6006 -v ~/Mask_RCNN/:/host waleedka/modern-deep-learning jupyter notebook --ip=0.0.0.0 --allow-root /host
