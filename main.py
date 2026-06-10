from utils.hurricane_intensity_evaluator import HurricaneIntensityEvaluator

if __name__ == "__main__":
    evaluator = HurricaneIntensityEvaluator("config/config_data_processing.yaml")
    evaluator.load_datasets()
    evaluator.evaluate_intensity_measures()
    evaluator.convert_points_to_polygons()
    evaluator.aggregate_intensity_measures()
    evaluator.interpolate_and_save_rasters()

    # Run "pf_evaluation" Jupyter notebook before calling this method!
    evaluator.calculate_failure_probabilities()

